"""配对核验实验核心。导入不加载模型、不读图片、不创建文件。

CPU 模型由 PairComparator 显式构造；调用 prepare/compare_prepared 可复用特征。
所有写入由调用者显式指定缓存/产物目录，规则文件不包含本机路径。
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

SAME_SOURCE_CODES = {"same_source_unchanged", "face_replaced", "face_edited_identity_kept", "edited_elsewhere"}
ALL_CODES = SAME_SOURCE_CODES | {"alignment_failed", "source_not_supported", "same_background_other_person", "inconclusive"}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_image(path: str | Path) -> np.ndarray:
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("图片无法解码")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError("图片无法编码")
    path.write_bytes(encoded.tobytes())


def rect_mask(shape: tuple[int, int], boxes: list | np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    for box in boxes:
        x1, y1, x2, y2 = np.asarray(box, dtype=float)[:4]
        x1, x2 = max(0, math.floor(x1)), min(width, math.ceil(x2))
        y1, y2 = max(0, math.floor(y1)), min(height, math.ceil(y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def expand_face(box: np.ndarray, factor: float) -> list[float]:
    x1, y1, x2, y2 = box[:4]
    dx, dy = (x2 - x1) * (factor - 1) / 2, (y2 - y1) * (factor - 1) / 2
    return [float(x1-dx), float(y1-dy), float(x2+dx), float(y2+dy)]


def person_box(box: np.ndarray, rule: dict) -> list[float]:
    x1, y1, x2, y2 = box[:4]
    width, height = x2-x1, y2-y1
    dx = width * (rule["person_width_factor"]-1) / 2
    return [float(x1-dx), float(y1-height*rule["person_top_extension"]),
            float(x2+dx), float(y2+height*rule["person_bottom_extension"])]


def bbox_iou(a: list | np.ndarray, b: list | np.ndarray) -> float:
    a, b = np.asarray(a), np.asarray(b)
    extent = np.maximum(0, np.minimum(a[2:4], b[2:4])-np.maximum(a[:2], b[:2]))
    intersection = float(extent[0]*extent[1])
    area_a = max(0, float(a[2]-a[0]))*max(0, float(a[3]-a[1]))
    area_b = max(0, float(b[2]-b[0]))*max(0, float(b[3]-b[1]))
    return intersection / max(area_a+area_b-intersection, 1e-9)


def map_box(box: list | np.ndarray, homography: np.ndarray) -> list[float]:
    x1, y1, x2, y2 = box[:4]
    points = np.float32([[[x1,y1], [x2,y1], [x2,y2], [x1,y2]]])
    mapped = cv2.perspectiveTransform(points, homography)[0]
    return [float(mapped[:,0].min()), float(mapped[:,1].min()),
            float(mapped[:,0].max()), float(mapped[:,1].max())]


def assign_faces(query_boxes: list, reference_boxes: list, minimum_iou: float) -> list[tuple[int, int, float]]:
    """一对一最大总 IoU 配对；低重合度不强行绑定。"""
    if not query_boxes or not reference_boxes:
        return []
    overlaps = np.array([[bbox_iou(a,b) for b in reference_boxes] for a in query_boxes])
    rows, cols = linear_sum_assignment(-overlaps)
    return [(int(i),int(j),float(overlaps[i,j])) for i,j in zip(rows,cols) if overlaps[i,j] >= minimum_iou]


@dataclass
class PreparedImage:
    image: np.ndarray
    original_shape: tuple[int, int]
    scale: float
    faces: list[dict]
    points: np.ndarray
    descriptors: np.ndarray
    prepare_seconds: float
    content_sha256: str
    cache_hit: bool = False


class PairComparator:
    def __init__(self, rule: dict | str | Path, model_dir: str | Path, cache_dir: str | Path | None = None):
        import onnxruntime as ort
        from insightface.model_zoo.model_zoo import ModelRouter
        self.rule = json.loads(Path(rule).read_text(encoding="utf-8-sig")) if isinstance(rule,(str,Path)) else dict(rule)
        self.model_dir = Path(model_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        cv2.setNumThreads(int(self.rule["cpu_threads"]))
        cv2.setRNGSeed(int(self.rule["seed"]))
        self.sift = cv2.SIFT_create(nfeatures=int(self.rule["sift_features"]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(self.rule["cpu_threads"])
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if self.rule.get("onnx_flush_denormals",True):
            options.add_session_config_entry("session.set_denormal_as_zero","1")
        start = time.perf_counter()
        # 直接读现有 ONNX，禁止自动下载和修改原环境。
        # 当前包的顶层 get_model 会丢弃 sess_options，直接调用路由器才能真正限制线程。
        self.detector = ModelRouter(str(self.model_dir/"det_10g.onnx")).get_model(providers=["CPUExecutionProvider"], sess_options=options)
        self.recognizer = ModelRouter(str(self.model_dir/"w600k_r50.onnx")).get_model(providers=["CPUExecutionProvider"], sess_options=options)
        self.detector.prepare(ctx_id=-1, input_size=(640,640), det_thresh=self.rule["face_detection_threshold"])
        self.recognizer.prepare(ctx_id=-1)
        self.model_load_seconds = time.perf_counter()-start
        prep_keys = ["max_image_side","sift_features","face_expansion","face_detection_threshold"]
        prep_config={k:self.rule[k] for k in prep_keys}
        prep_config.update(cpu_threads=self.rule["cpu_threads"],session_construction="explicit_modelrouter_v2",onnx_flush_denormals=self.rule.get("onnx_flush_denormals",True))
        self.preparation_key = hashlib.sha256(json.dumps(prep_config,sort_keys=True).encode()).hexdigest()[:16]

    def prepare(self, image_or_path: np.ndarray | str | Path) -> PreparedImage:
        from insightface.app.common import Face
        start = time.perf_counter()
        if isinstance(image_or_path,(str,Path)):
            digest = file_sha256(image_or_path)
            image = None
        else:
            image = image_or_path
            digest = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        cache = self.cache_dir/f"{digest}_{self.preparation_key}.npz" if self.cache_dir else None
        if cache and cache.exists():
            with np.load(cache,allow_pickle=False) as data:
                metadata=json.loads(str(data["metadata"]))
                faces=metadata["faces"]
                embeddings=data["embeddings"]
                for i, face in enumerate(faces):
                    face["embedding"]=embeddings[i]
                return PreparedImage(data["image"],tuple(metadata["original_shape"]),metadata["scale"],faces,
                                     data["points"],data["descriptors"],metadata["prepare_seconds"],digest,True)
        if image is None:
            image=read_image(image_or_path)
        original_shape=image.shape[:2]
        scale=min(1.0,self.rule["max_image_side"]/max(original_shape))
        if scale<1:
            image=cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)),interpolation=cv2.INTER_AREA)
        boxes,kps=self.detector.detect(image,max_num=0)
        faces=[]
        for index, box in enumerate(boxes):
            face=Face(bbox=box[:4],kps=kps[index] if kps is not None else None,det_score=float(box[4]))
            self.recognizer.get(image,face)
            vector=np.asarray(face.normed_embedding,dtype=np.float32)
            faces.append({"face_id":index,"bbox":box[:4].astype(float).tolist(),"det_score":float(box[4]),
                          "kps":face.kps.astype(float).tolist() if face.kps is not None else None,"embedding":vector})
        mask=255-rect_mask(image.shape[:2],[expand_face(np.asarray(f["bbox"]),self.rule["face_expansion"]) for f in faces])
        keypoints,descriptors=self.sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
        points=np.float32([keypoint.pt for keypoint in keypoints]).reshape(-1,2)
        if descriptors is None:
            descriptors=np.empty((0,128),np.float32)
        elapsed=time.perf_counter()-start
        prepared=PreparedImage(image,original_shape,scale,faces,points,descriptors,elapsed,digest)
        if cache:
            metadata={"original_shape":original_shape,"scale":scale,"prepare_seconds":elapsed,
                      "faces":[{k:v for k,v in f.items() if k!="embedding"} for f in faces]}
            np.savez_compressed(cache,image=image,points=points,descriptors=descriptors,
                                embeddings=np.asarray([f["embedding"] for f in faces],dtype=np.float32).reshape(-1,512),
                                metadata=np.array(json.dumps(metadata,ensure_ascii=False)))
        return prepared

    def compare(self, query: np.ndarray | str | Path, reference: np.ndarray | str | Path, *, return_artifacts: bool=False):
        start=time.perf_counter()
        q=self.prepare(query)
        r=self.prepare(reference)
        result,artifacts=self.compare_prepared(q,r,return_artifacts=return_artifacts)
        result["elapsed_wall_seconds"]=time.perf_counter()-start
        return result,artifacts

    def compare_prepared(self,q:PreparedImage,r:PreparedImage,*,return_artifacts:bool=False):
        start=time.perf_counter()
        rule=self.rule
        result={"code":"inconclusive","reason":"尚未完成", "source_supported":False,"other_edits":False,
                "query_face_count":len(q.faces),"reference_face_count":len(r.faces),"faces":[],
                "alignment":{"inliers":0,"good_matches":0},"metrics":{},
                "query_sha256":q.content_sha256,"reference_sha256":r.content_sha256}
        result["query_faces"]=[{"face_id":f["face_id"],"bbox":f["bbox"],"bbox_original":(np.array(f["bbox"])/q.scale).tolist(),"det_score":f["det_score"]} for f in q.faces]
        result["reference_faces"]=[{"face_id":f["face_id"],"bbox":f["bbox"],"bbox_original":(np.array(f["bbox"])/r.scale).tolist(),"det_score":f["det_score"]} for f in r.faces]
        result["faces"]=[{"query_face_id":f["face_id"],"query_bbox":f["bbox"],"query_bbox_original":(np.array(f["bbox"])/q.scale).tolist(),
                          "status":"not_compared","cosine_similarity":None,"reason":"尚未建立可比对的同源关系"} for f in q.faces]
        artifacts={"query":q.image,"reference":r.image} if return_artifacts else {}
        def finish(code,reason):
            result["code"]=code
            result["reason"]=reason
            result["source_supported"]=code in SAME_SOURCE_CODES
            result["comparison_seconds"]=time.perf_counter()-start
            result["uncached_pair_seconds"]=result["comparison_seconds"]+q.prepare_seconds+r.prepare_seconds
            result["preparation_seconds"]={"query":q.prepare_seconds,"reference":r.prepare_seconds}
            result["cache_hits"]={"query":q.cache_hit,"reference":r.cache_hit}
            return result,artifacts
        if len(q.descriptors)<2 or len(r.descriptors)<2:
            return finish("alignment_failed","非脸部区域缺少可匹配特征")
        raw=cv2.BFMatcher(cv2.NORM_L2).knnMatch(q.descriptors,r.descriptors,k=2)
        good=[m for pair in raw if len(pair)==2 for m,n in [pair] if m.distance<rule["lowe_ratio"]*n.distance]
        result["alignment"]["good_matches"]=len(good)
        if len(good)<rule["min_inliers"]:
            return finish("alignment_failed","Lowe 比值筛选后匹配不足")
        src=np.float32([q.points[m.queryIdx] for m in good]).reshape(-1,1,2)
        dst=np.float32([r.points[m.trainIdx] for m in good]).reshape(-1,1,2)
        cv2.setRNGSeed(int(rule["seed"]))
        homography,inliers=cv2.findHomography(src,dst,cv2.RANSAC,rule["ransac_threshold"])
        if homography is None or inliers is None or not np.isfinite(homography).all():
            return finish("alignment_failed","未求得有效单应矩阵")
        inlier_mask=inliers.ravel().astype(bool)
        count=int(inlier_mask.sum())
        ratio=count/max(len(good),1)
        hull_fraction=float(cv2.contourArea(cv2.convexHull(dst[inlier_mask]))/max(r.image.shape[0]*r.image.shape[1],1)) if count>=3 else 0.0
        result["alignment"].update(inliers=count,inlier_ratio=ratio,inlier_hull_fraction=hull_fraction,homography=homography.tolist())
        if count<rule["min_inliers"] or ratio<rule["min_inlier_ratio"] or hull_fraction<rule["min_inlier_hull_fraction"]:
            return finish("alignment_failed","内点数量、比例或空间覆盖不足")
        height,width=r.image.shape[:2]
        corners=cv2.perspectiveTransform(np.float32([[[0,0],[q.image.shape[1],0],[q.image.shape[1],q.image.shape[0]],[0,q.image.shape[0]]]]),homography)[0]
        area_ratio=abs(float(cv2.contourArea(corners)))/(height*width)
        if not cv2.isContourConvex(corners) or not rule["min_warp_area_ratio"]<=area_ratio<=rule["max_warp_area_ratio"]:
            return finish("alignment_failed","映射轮廓退化或面积异常")
        warped=cv2.warpPerspective(q.image,homography,(width,height),flags=cv2.INTER_LINEAR)
        valid=cv2.warpPerspective(np.full(q.image.shape[:2],255,np.uint8),homography,(width,height),flags=cv2.INTER_NEAREST)
        valid=cv2.erode(valid,np.ones((7,7),np.uint8))>0
        result["alignment"]["valid_fraction"]=float(valid.mean())
        if valid.mean()<rule["min_valid_fraction"]:
            return finish("alignment_failed","有效重叠区域不足")
        mapped_boxes=[map_box(face["bbox"],homography) for face in q.faces]
        all_face_boxes=[expand_face(np.asarray(box),rule["face_expansion"]) for box in mapped_boxes]+[
            expand_face(np.asarray(face["bbox"]),rule["face_expansion"]) for face in r.faces]
        all_person_boxes=[person_box(np.asarray(box),rule) for box in mapped_boxes]+[
            person_box(np.asarray(face["bbox"]),rule) for face in r.faces]
        face_union=rect_mask((height,width),all_face_boxes)>0
        person_union=rect_mask((height,width),all_person_boxes)>0
        background=valid&~person_union
        result["metrics"]["background_pixels"]=int(background.sum())
        result["metrics"]["background_fraction"]=float(background.mean())
        if int(background.sum())<rule["background_min_pixels"] or background.mean()<rule["background_min_fraction"]:
            return finish("inconclusive","人物以外的有效背景不足，无法稳健核验同源")
        normalized=warped.astype(np.float32)
        color_parameters=[]
        for channel in range(3):
            qa=warped[:,:,channel][background].astype(np.float32)
            rb=r.image[:,:,channel][background].astype(np.float32)
            qlo,qhi=np.percentile(qa,[10,90]); rlo,rhi=np.percentile(rb,[10,90])
            gain=float(np.clip((rhi-rlo)/max(qhi-qlo,10),rule["color_gain_min"],rule["color_gain_max"]))
            offset=float(np.clip(np.median(rb-gain*qa),-rule["color_offset_limit"],rule["color_offset_limit"]))
            normalized[:,:,channel]=np.clip(normalized[:,:,channel]*gain+offset,0,255)
            color_parameters.append({"gain":gain,"offset":offset})
        normalized=normalized.astype(np.uint8)
        a=cv2.GaussianBlur(normalized,(5,5),0.8)
        b=cv2.GaussianBlur(r.image,(5,5),0.8)
        absolute=np.abs(a.astype(np.float32)-b.astype(np.float32)).mean(axis=2)/255.0
        _,ssim_map=structural_similarity(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY),full=True,data_range=255)
        ssim_map=ssim_map.astype(np.float32)
        change=(absolute>rule["change_abs_threshold"])&((1-ssim_map)>rule["change_ssim_loss_threshold"])&valid
        result["metrics"]["color_normalization"]=color_parameters
        # 横向局部文字条单列为其他改动；不把文字覆盖的人物周边当成换了整个人。
        # 不推断文字内容，且移除后仍须满足剩余背景和人物周边的像素要求。
        overlay=np.zeros((height,width),bool)
        strip_boxes=[]
        _,_,components,_=cv2.connectedComponentsWithStats(change.astype(np.uint8),8)
        for x,y,bw,bh,area in components[1:]:
            if bw/width>=rule["overlay_strip_min_width_fraction"] and bh/height<=rule["overlay_strip_max_height_fraction"] and area/max(bw*bh,1)>=rule["overlay_strip_min_fill_fraction"]:
                overlay[y:y+bh,x:x+bw]=True
                strip_boxes.append([int(x),int(y),int(x+bw),int(y+bh)])
        result["metrics"]["local_overlay_rectangles"]=strip_boxes
        comparison_background=background&~overlay
        if int(comparison_background.sum())<rule["background_min_pixels"] or comparison_background.mean()<rule["background_min_fraction"]:
            return finish("inconclusive","扣除局部覆盖条之后，可靠背景不足")
        def region_stats(mask):
            n=int(mask.sum())
            if not n:
                return {"pixels":0,"changed_fraction":None,"ssim_median":None,"mae_median":None,"mae_mean":None}
            return {"pixels":n,"changed_fraction":float(change[mask].mean()),"ssim_median":float(np.median(ssim_map[mask])),
                    "mae_median":float(np.median(absolute[mask])),"mae_mean":float(absolute[mask].mean())}
        bg=region_stats(comparison_background)
        result["metrics"]["background"]=bg
        result["metrics"]["face_change_concentration"]=float(change[face_union].sum()/max(change.sum(),1))
        if return_artifacts:
            artifacts.update(aligned=normalized,reference=r.image,query=q.image,absolute=absolute,change_mask=change,
                             valid_mask=valid,background_mask=background,face_mask=face_union,person_mask=person_union)
        if bg["changed_fraction"]>rule["background_changed_fraction_max"] or bg["ssim_median"]<rule["background_ssim_median_min"] or bg["mae_median"]>rule["background_mae_median_max"]:
            return finish("source_not_supported","已完成几何对齐，但人物以外的背景整体不一致")
        outside=change&background
        component_count,component_labels,component_stats,_=cv2.connectedComponentsWithStats(outside.astype(np.uint8),8)
        largest_component=int(component_stats[1:,cv2.CC_STAT_AREA].max()) if component_count>1 else 0
        result["other_edits"]=bool(strip_boxes or (bg["changed_fraction"]>=rule["other_edit_fraction_min"] and largest_component/(height*width)>=rule["other_edit_component_fraction_min"]))
        result["metrics"]["other_edit_largest_component_pixels"]=largest_component
        result["faces"]=[]
        assignments=assign_faces(mapped_boxes,[f["bbox"] for f in r.faces],rule["face_match_iou"])
        by_query={i:(j,iou) for i,j,iou in assignments}
        matched_references={j for _,j,_ in assignments}
        states=[]
        for index,face in enumerate(q.faces):
            row={"query_face_id":index,"query_bbox":face["bbox"],"query_bbox_original":(np.array(face["bbox"])/q.scale).tolist(),
                 "mapped_query_bbox":mapped_boxes[index],"status":"unmatched","cosine_similarity":None,
                 "query_det_score":face["det_score"]}
            if index not in by_query:
                row["reason"]="候选图对应位置没有可一对一绑定的人脸"
                result["faces"].append(row); states.append("unknown"); continue
            j,iou=by_query[index]
            ref=r.faces[j]
            cosine=float(np.clip(np.dot(face["embedding"],ref["embedding"]),-1,1))
            region=rect_mask((height,width),[expand_face(np.asarray(mapped_boxes[index]),rule["face_expansion"]),expand_face(np.asarray(ref["bbox"]),rule["face_expansion"])])>0
            person=rect_mask((height,width),[person_box(np.asarray(mapped_boxes[index]),rule),person_box(np.asarray(ref["bbox"]),rule)])>0
            fs=region_stats(region&valid)
            ps=region_stats(person&~face_union&valid&~overlay)
            row.update(reference_face_id=j,reference_bbox=ref["bbox"],reference_bbox_original=(np.array(ref["bbox"])/r.scale).tolist(),
                       match_iou=iou,cosine_similarity=cosine,face_region=fs,surround_region=ps,reference_det_score=ref["det_score"])
            box=np.array(face["bbox"]); rb=np.array(ref["bbox"])
            if min(box[2]-box[0],box[3]-box[1],rb[2]-rb[0],rb[3]-rb[1])<rule["face_min_side"] or ps["pixels"]<rule["surround_min_pixels"]:
                state="unknown"; row["reason"]="脸部尺寸或人物周边有效像素不足"
            else:
                face_changed=fs["changed_fraction"]>=rule["face_changed_fraction_min"]
                surround_stable=ps["changed_fraction"]<=rule["surround_changed_fraction_max"] and ps["mae_median"]<=rule["surround_mae_median_max"]
                if cosine<=rule["identity_different_max"]:
                    if not face_changed:
                        state="unknown"; row["reason"]="身份相似度低但未见足够脸部像素变化"
                    elif surround_stable:
                        state="face_replaced"; row["reason"]="身份不一致且人物周边基本一致"
                    else:
                        state="same_background_other_person"; row["reason"]="身份与人物周边均明显不同"
                elif cosine>=rule["identity_same_min"]:
                    state="face_edited_identity_kept" if face_changed else "unchanged"
                    row["reason"]="身份一致；脸部像素已改变" if face_changed else "身份和脸部像素基本一致"
                    if not surround_stable:
                        row["surround_changed"]=True
                        result["other_edits"]=True
                else:
                    state="unknown"; row["reason"]="身份相似度位于预留灰区"
            row["status"]=state
            states.append(state)
            result["faces"].append(row)
        for index,ref in enumerate(r.faces):
            if index not in matched_references:
                result["faces"].append({"query_face_id":None,"reference_face_id":index,"reference_bbox":ref["bbox"],"status":"unmatched", "cosine_similarity":None,
                                        "reason":"候选图人脸未与查询图建立一对一绑定"})
                states.append("unknown")
        if "face_replaced" in states and "same_background_other_person" in states:
            return finish("inconclusive","不同人脸支持的修改机制不一致")
        if "face_replaced" in states:
            return finish("face_replaced","至少一张已绑定人脸的身份不同，且其人物周边基本一致")
        if "same_background_other_person" in states:
            return finish("same_background_other_person","背景接近，但至少一处人物整体不同")
        if "unknown" in states:
            return finish("inconclusive","至少一张人脸无法可靠比较")
        if "face_edited_identity_kept" in states:
            return finish("face_edited_identity_kept","同源证据成立，脸部有变化但身份相似度保持一致")
        if result["other_edits"]:
            return finish("edited_elsewhere","同源证据成立，检出的改动位于脸部之外")
        return finish("same_source_unchanged","同源证据成立，未见脸部改动；无人脸时仅代表可见区域一致")


def save_artifacts(output_dir: str | Path, artifacts: dict, result: dict) -> dict:
    """仅在调用者要求时输出可视化；不渲染模型身份解释。"""
    directory=Path(output_dir)
    directory.mkdir(parents=True,exist_ok=True)
    paths={}
    for name in ("aligned","reference","query"):
        if name in artifacts:
            path=directory/f"{name}.png"
            write_image(path,artifacts[name]); paths[name]=str(path)
    if "absolute" in artifacts:
        heat=cv2.applyColorMap(np.uint8(np.clip(artifacts["absolute"]*4,0,1)*255),cv2.COLORMAP_JET)
        heat[~artifacts["valid_mask"]]=0
        overlay=cv2.addWeighted(artifacts["reference"],0.55,heat,0.45,0)
        write_image(directory/"difference.png",overlay)
        write_image(directory/"change_mask.png",artifacts["change_mask"].astype(np.uint8)*255)
        paths.update(difference=str(directory/"difference.png"),change_mask=str(directory/"change_mask.png"))
    (directory/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return paths
