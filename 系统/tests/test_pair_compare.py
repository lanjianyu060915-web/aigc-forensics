"""不加载模型的算法行为检验；用合成纹理验证状态边界与一对一绑定。"""
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from modules.pair_compare.pair_compare import PreparedImage, PairComparator, assign_faces, rect_mask


class PairRulesTest(unittest.TestCase):
    def setUp(self):
        self.rule=json.loads((Path(__file__).resolve().parents[1]/"modules/pair_compare/rule.json").read_text(encoding="utf-8"))
        self.engine=PairComparator.__new__(PairComparator)
        self.engine.rule=self.rule
        rng=np.random.default_rng(445)
        self.base=cv2.GaussianBlur(rng.integers(0,256,(384,512,3),dtype=np.uint8),(5,5),0.7)
        self.box=[195,70,255,145]

    def prepared(self,image,face=True,different=False,points=True):
        rng=np.random.default_rng(555)
        xy=np.array([(x,y) for y in range(25,380,40) for x in range(20,500,40)],np.float32)
        descriptors=rng.random((len(xy),128),dtype=np.float32)*100
        if not points:
            xy=np.empty((0,2),np.float32);descriptors=np.empty((0,128),np.float32)
        embedding=np.zeros(512,np.float32);embedding[int(different)]=1
        faces=[{"face_id":0,"bbox":self.box,"embedding":embedding,"det_score":.99}] if face else []
        return PreparedImage(image,image.shape[:2],1.0,faces,xy,descriptors,0,"synthetic")

    def classify(self,image,*,face=True,different=False):
        result,_=self.engine.compare_prepared(self.prepared(image,face,different),self.prepared(self.base,face),return_artifacts=True)
        json.dumps(result)
        return result

    def test_one_to_one_face_assignment(self):
        matches=assign_faces([[0,0,10,10],[1,0,11,10]],[[0,0,10,10]],.25)
        self.assertEqual(len(matches),1)

    def test_alignment_failure_not_distinct(self):
        result,_=self.engine.compare_prepared(self.prepared(self.base,points=False),self.prepared(self.base))
        self.assertEqual(result["code"],"alignment_failed")
        self.assertFalse(result["source_supported"])

    def test_no_face_copy(self):
        self.assertEqual(self.classify(self.base,face=False)["code"],"same_source_unchanged")

    def test_unchanged_face(self):
        self.assertEqual(self.classify(self.base)["code"],"same_source_unchanged")

    def test_identity_difference_without_pixel_change_refuses(self):
        self.assertEqual(self.classify(self.base,different=True)["code"],"inconclusive")

    def test_face_swap_has_joint_conditions(self):
        image=self.base.copy();image[70:145,195:255]=[210,20,30]
        self.assertEqual(self.classify(image,different=True)["code"],"face_replaced")

    def test_identity_preserving_edit(self):
        image=self.base.copy();image[70:145,195:255]=[210,20,30]
        self.assertEqual(self.classify(image)["code"],"face_edited_identity_kept")

    def test_face_swap_with_caption_strip(self):
        image=self.base.copy();image[70:145,195:255]=[210,20,30];image[355:384,:]=245
        result=self.classify(image,different=True)
        self.assertEqual(result["code"],"face_replaced")
        self.assertTrue(result["other_edits"])

    def test_other_person_has_body_difference(self):
        image=self.base.copy();image[50:295,150:300]=[210,20,30]
        self.assertEqual(self.classify(image,different=True)["code"],"same_background_other_person")

    def test_background_edit_not_swap(self):
        image=self.base.copy();image[20:60,350:450]=[210,20,30]
        result=self.classify(image)
        self.assertEqual(result["code"],"edited_elsewhere")
        self.assertTrue(result["other_edits"])

    def test_matching_geometry_but_wrong_pixels(self):
        image=255-self.base
        self.assertEqual(self.classify(image)["code"],"source_not_supported")


if __name__=="__main__":unittest.main(verbosity=2)
