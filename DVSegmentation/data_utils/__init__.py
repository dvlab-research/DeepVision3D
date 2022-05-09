from .ScanNetDataLoader import MyDataset as ScanNetDataset

__dataset__all__ = {
    'scannetv2': ScanNetDataset,
}