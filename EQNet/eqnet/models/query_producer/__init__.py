from .qnet import PCDetQNetNeck, MMDet3DQNetNeck, DVClsQNetNeck, DVSegQNetNeck

# pcdet.
__pcdet_all__ = {
    'QNetNeck': PCDetQNetNeck,
}

# mmdet3d.
__mmdet3d_all__ = {
    'MMDet3DQNetNeck': MMDet3DQNetNeck,
}

# dv_cls.
__dvcls__all__ = {
    'QNetNeck': DVClsQNetNeck,
}

# dv_seg.
__dvseg__all__ = {
    'QNetNeck': DVSegQNetNeck,
}