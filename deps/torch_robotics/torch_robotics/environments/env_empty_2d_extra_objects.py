import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.autograd.functional import jacobian

from torch_robotics.environments import EnvEmpty2D
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvEmpty2DExtraObjects(EnvEmpty2D):

    def __init__(self, tensor_args=None, **kwargs):
        obj_extra_list = [
            # MultiSphereField(
            #     # Large obstacles in the middle - more challenging navigation
            #     np.array([[0.0, 0.0]]),  
            #     np.array([0.3]),  # much larger spheres
            #     tensor_args=tensor_args
            # ),
            MultiBoxField(
                np.array(  # (n, 2) array of box centers.
                    [
                        [0.0, 0.0], [0.0, 0.6], [0.0, -0.6],
                    ]
                ),
                np.array(  # (n, 2) array of box sizes.
                    [
                        [0.2, 0.5],
                        [0.2, 0.38],
                        [0.2, 0.38],
                    ]
                ),
                tensor_args=tensor_args
            )
        ]

        super().__init__(
            name=self.__class__.__name__,
            obj_extra_list=[ObjectField(obj_extra_list, 'empty2d-extraobjects')],
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    import os
    from pathlib import Path
    
    # Get mmd root directory and create media subdirectory
    mmd_root = Path(__file__).resolve().parents[4]  # Navigate to /home/.../mmd/
    output_dir = mmd_root / 'media'
    output_dir.mkdir(exist_ok=True)
    
    env = EnvEmpty2DExtraObjects(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    
    # Save environment rendering
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    output_path = output_dir / 'env_empty_2d_extra_objects.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

    # Save SDF and gradient
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)
    env.render_grad_sdf(ax, fig)
    output_path = output_dir / 'env_empty_2d_sdf_gradient.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
