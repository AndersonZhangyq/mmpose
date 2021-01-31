from torchinfo import summary

from mmpose.models.backbones.evopose2d import EvoPose2D


def test_evopose2d_backbone():
    args = {
        'device': 'cpu',
        'depth': 10,
        'col_names': ['kernel_size', 'output_size', 'num_params', 'mult_adds']
    }
    # se_module = SEModule(40, 40, 0.25, "block1a_", "swish")
    # summary(se_module, (1, 40, 192, 144), **args)
    # block = nn.Sequential(
    #     nn.Sequential(Block(40)))
    # summary(block, (1, 40, 192, 144), **args)
    model = EvoPose2D(
        genotype=[[3, 1, 2, 1], [3, 3, 3, 2], [5, 2, 5, 2], [3, 4, 10, 2],
                  [5, 2, 14, 1], [5, 4, 16, 1], [3, 2, 10, 1]])

    # model(torch.randn(2, 3, 384, 288))
    summary(model, (1, 3, 384, 288), **args)


if __name__ == '__main__':
    test_evopose2d_backbone()
