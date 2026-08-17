import torch

from mri_lgformer import MRILGFormerT1


def test_default_parameter_count():
    model = MRILGFormerT1()
    assert sum(parameter.numel() for parameter in model.parameters()) == 24_897_808


def test_state_dict_round_trip():
    model = MRILGFormerT1(
        dim=8,
        num_blocks=(1, 2, 1, 1),
        num_refinement_blocks=1,
        heads=(1, 1, 2, 4),
    )
    clone = MRILGFormerT1(
        dim=8,
        num_blocks=(1, 2, 1, 1),
        num_refinement_blocks=1,
        heads=(1, 1, 2, 4),
    )
    clone.load_state_dict(model.state_dict(), strict=True)
    sample = torch.randn(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(sample), clone(sample))
