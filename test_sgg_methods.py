"""
1. STTran
2. DsgDetr
3. Tempura
"""
from lib.supervised.config import Config
from test_sgg_base import TestSGGBase


class TestSTTran(TestSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sgg.sttran.sttran import STTran
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._test_dataset.attention_relationships),
                             spatial_class_num=len(self._test_dataset.spatial_relationships),
                             contact_class_num=len(self._test_dataset.contacting_relationships),
                             obj_classes=self._test_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

    def process_test_video(self, video_entry, frame_size, gt_annotation) -> dict:
        pred = self._model(video_entry)
        return pred


class TestDsgDetr(TestSGGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.dsgdetr import DsgDETR
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher

        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._test_dataset.attention_relationships),
                              spatial_class_num=len(self._test_dataset.spatial_relationships),
                              contact_class_num=len(self._test_dataset.contacting_relationships),
                              obj_classes=self._test_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._matcher.eval()

    def process_test_video(self, video_entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(video_entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        prediction = self._model(video_entry)
        return prediction


class TestTempura(TestSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sgg.tempura.tempura import TEMPURA
        model = TEMPURA(mode=self._conf.mode,
                        attention_class_num=len(self._test_dataset.attention_relationships),
                        spatial_class_num=len(self._test_dataset.spatial_relationships),
                        contact_class_num=len(self._test_dataset.contacting_relationships),
                        obj_classes=self._test_dataset.object_classes,
                        enc_layer_num=self._conf.enc_layer,
                        dec_layer_num=self._conf.dec_layer,
                        obj_mem_compute=self._conf.obj_mem_compute,
                        rel_mem_compute=self._conf.rel_mem_compute,
                        take_obj_mem_feat=self._conf.take_obj_mem_feat,
                        mem_fusion=self._conf.mem_fusion,
                        selection=self._conf.mem_feat_selection,
                        selection_lambda=self._conf.mem_feat_lambda,
                        obj_head=self._conf.obj_head,
                        rel_head=self._conf.rel_head,
                        K=self._conf.K,
                        tracking=self._conf.tracking).to(device=self._device)
        model.eval()

    def process_test_video(self, video_entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.tempura.ds_track import get_sequence
        if self._conf.tracking:
            get_sequence(video_entry, gt_annotation, frame_size, self._conf.mode)
        prediction = self._model(video_entry)
        return prediction


def main():
    conf = Config()
    if conf.method_name == "sttran":
        evaluate_class = TestSTTran(conf)
    elif conf.method_name == "dsgdetr":
        evaluate_class = TestDsgDetr(conf)
    elif conf.method_name == "tempura":
        evaluate_class = TestTempura(conf)
    else:
        raise NotImplementedError

    evaluate_class.init_method_evaluation()


if __name__ == "__main__":
    main()
