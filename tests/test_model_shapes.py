import pytest
import torch

from mri_lgformer import MRILGFormerT1


@pytest.mark.parametrize("allocation", ["proposed", "all_local", "reversed"])
def test_output_shape(allocation):
    model = MRILGFormerT1(
        dim=8,
        num_blocks=(1, 2, 1, 1),
        num_refinement_blocks=1,
        heads=(1, 1, 2, 4),
        allocation=allocation,
    ).eval()
    sample = torch.randn(1, 1, 32, 40)
    with torch.no_grad():
        output = model(sample)
    assert output.shape == sample.shape
    assert torch.isfinite(output).all()


def test_rejects_non_divisible_shape():
    model = MRILGFormerT1(
        dim=8,
        num_blocks=(1, 2, 1, 1),
        num_refinement_blocks=1,
        heads=(1, 1, 2, 4),
    )
    with pytest.raises(ValueError, match="divisible by 8"):
        model(torch.randn(1, 1, 31, 40))
