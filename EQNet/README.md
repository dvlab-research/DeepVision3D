# Support producer & Query producer networks

This maintains the unified implementation of different support producer and query producer networks which can be easily integrated
into downstream models for different codebases.

## Support Producer Networks
|                                             |paper| implementation | 
|---------------------------------------------|:----------:|:-------:|
| SparseConvNet |[link](https://arxiv.org/pdf/1706.01307.pdf)| [code](eqnet/models/support_producer/spconv_backbone.py) |
| PointNet++ | [link](https://arxiv.org/pdf/1706.02413.pdf) | [code](eqnet/models/support_producer/pointnet2_backbone.py) |

## Query Producer Networks
|                                             |paper| implementation | 
|---------------------------------------------|:----------:|:-------:|
| Q-Net |[link](https://arxiv.org/pdf/2203.01252.pdf)| [code](eqnet/models/query_producer/qnet.py) |
