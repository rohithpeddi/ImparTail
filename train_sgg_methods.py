from lib.supervised.config import Config
from train_sgg_base import TrainSGGBase
from lib.supervised.sgg.sttran.sttran import STTran
from lib.supervised.sgg.dsgdetr.dsgdetr import DsgDETR
from lib.supervised.sgg.tempura.tempura import TEMPURA

class TrainSTTran(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred


class TrainDsgDetr(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):

        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher

        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred


class TrainSTTranSTL(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._enable_stl_loss = True

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred


class TrainDsgDetrSTL(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._enable_stl_loss = True

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred


def main():
    conf = Config()
    if conf.method_name == "sttran":
        train_class = TrainSTTran(conf)
    elif conf.method_name == "dsgdetr":
        train_class = TrainDsgDetr(conf)
    elif conf.method_name == "sttran_stl":
        train_class = TrainSTTranSTL(conf)
    elif conf.method_name == "dsgdetr_stl":
        train_class = TrainDsgDetrSTL(conf)
    else:
        raise NotImplementedError

    train_class.init_method_training()


if __name__ == "__main__":
    main()
