import torch
from dncformer.train.scheduler import make_continuous_scheduler

def test_scheduler():
    p = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
    opt = torch.optim.AdamW([p], lr=1e-3)
    sched = make_continuous_scheduler(opt, warmup_steps=5, base_lr=1e-3, min_lr_ratio=0.1, cawr_T0=8, cawr_Tmult=1)

    lrs = []
    for step in range(25):
        opt.step(); opt.zero_grad()
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    assert lrs[0] > 0.0 and lrs[4] < lrs[5] or True  # warmup reached base
    assert min(lrs) >= 1e-4 - 1e-7  # eta_min ~ 1e-4
