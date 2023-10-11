import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from scipy import interpolate


class RoomSimulation:
    def __init__(self,
                 corners: np.ndarray,
                 upsample_resolution: int = None,
                 ):

        if upsample_resolution is None:
            upsample_resolution = 200 // len(corners)
        points = interpolate.interp1d(
            np.linspace(0, 1, len(corners)),
            corners,
            axis=0,
        )(np.linspace(0, 1, upsample_resolution * (len(corners) - 1) + 1))
        self.points = points

        vortex_points = points[:-1, :]
        collocation_points = (points[:-1, :] + points[1:, :]) / 2

        opti = asb.Opti()
        strengths = opti.variable(init_guess=np.zeros(len(vortex_points)))

        from potentialflowvisualizer import Flowfield, Doublet, Vortex

        fans = [
            Doublet(
                strength=0.5,
                x=4,
                y=2,
                alpha=np.radians(-20),  # Angle of the doublet axis
            ),
            Doublet(
                strength=0.5,
                x=-2,
                y=2,
                alpha=np.radians(40),  # Angle of the doublet axis
            ),
            Doublet(
                strength=0.5,
                x=0,
                y=-3,
                alpha=np.radians(170),  # Angle of the doublet axis
            )
        ]

        field = Flowfield(objects=fans + [
            Vortex(
                strength=strengths[i],
                x=vortex_points[i, 0],
                y=vortex_points[i, 1],
            )
            for i in range(len(vortex_points))
        ])

        V_collocations = np.stack([
            field.get_x_velocity_at(collocation_points),
            field.get_y_velocity_at(collocation_points),
        ], axis=1)

        tangent_vectors = np.roll(collocation_points, 1, axis=0) - np.roll(collocation_points, -1, axis=0)
        normal_vectors = np.stack([
            tangent_vectors[:, 1],
            -tangent_vectors[:, 0]
        ], axis=1)

        wall_normal_velocity_at_collocation_points = np.sum(
            V_collocations * normal_vectors,
            axis=1
        )

        opti.subject_to(np.sum(strengths) == 0)
        opti.minimize(np.sum(wall_normal_velocity_at_collocation_points ** 2))

        sol = opti.solve()

        self.corners = corners
        self.points = points
        self.field: Flowfield = sol(field)
        self.fans = sol(fans)

    def get_velocity(self,
                     x: float | np.ndarray,
                     y: float | np.ndarray
                     ) -> tuple[float | np.ndarray, float | np.ndarray]:
        points = np.stack([
            np.reshape(x, -1),
            np.reshape(y, -1),
        ], axis=1)
        u = np.reshape(self.field.get_x_velocity_at(points), np.shape(x))
        v = np.reshape(self.field.get_y_velocity_at(points), np.shape(y))

        return u, v

    def point_is_inside_room(self,
                             x: float | np.ndarray,
                             y: float | np.ndarray
                             ) -> bool | np.ndarray:
        points = np.stack([
            np.reshape(x, -1),
            np.reshape(y, -1),
        ], axis=1)

        from matplotlib.path import Path
        is_inside_room = Path(self.corners).contains_points(points)

        return np.reshape(is_inside_room, np.shape(x))

    def plot_room(self, show=True, set_equal=True):
        plt.plot(*self.corners.T, "-k", linewidth=3)
        if set_equal:
            p.equal()
        if show:
            p.show_plot(None, "$x$ [m]", "$y$ [m]")

    def plot_flowfield(self, show=True, set_equal=True):
        xlim = np.array([self.points[:, 0].min(), self.points[:, 0].max()])
        ylim = np.array([self.points[:, 1].min(), self.points[:, 1].max()])

        from matplotlib.colors import LinearSegmentedColormap, LogNorm
        from matplotlib import cm

        # Extract colors from RdBu and coolwarm colormaps
        color_blue = "royalblue"  # "#cm.RdBu(float(0))
        color_gray = "darkgray"
        color_red = "crimson"  # cm.RdBu(float(1))

        # Create custom colormap using LinearSegmentedColormap.from_list
        colors = [color_blue, color_gray, color_red]
        positions = [1e-1, 1e-2, 1e-3]
        cmap = LinearSegmentedColormap.from_list('Custom_Colormap', colors, N=256)
        cmap.set_under(plt.gca().get_facecolor())

        # Logarithmic normalization
        norm = LogNorm(vmin=1e-3, vmax=1)

        fig, ax = plt.subplots()

        X, Y = np.meshgrid(
            np.linspace(*xlim, 300),
            np.linspace(*ylim, 300),
        )
        U, V = self.get_velocity(X, Y)
        velmag = np.sqrt(U ** 2 + V ** 2)

        inside_room = self.point_is_inside_room(X, Y)
        U = np.where(
            inside_room,
            U,
            np.nan
        )
        V = np.where(
            inside_room,
            V,
            np.nan
        )

        plt.streamplot(
            X, Y, U, V,
            color=velmag,
            linewidth=1,
            minlength=0.02,
            cmap=cmap,
            norm=norm,
            broken_streamlines=False
        )
        plt.colorbar(label="Velocity Magnitude [m/s]")

        plt.plot(*self.points.T, "-k", linewidth=3)

        for fan in self.fans:
            ax.add_patch(
                plt.Circle(
                    xy=(fan.x, fan.y),
                    radius=0.3,
                    color="k",
                    zorder=4,
                )
            )
            ax.add_patch(
                plt.arrow(
                    x=fan.x - 0.2 * np.cos(fan.alpha),
                    y=fan.y - 0.2 * np.sin(fan.alpha),
                    dx=0.2 * np.cos(fan.alpha),
                    dy=0.2 * np.sin(fan.alpha),
                    width=0.02,
                    head_width=0.1,
                    color="w",
                    zorder=5,
                )
            )

        plt.xlim(xlim + np.array([-0.1, 0.1]))
        plt.ylim(ylim + np.array([-0.1, 0.1]))

        if set_equal:
            p.equal()
        if show:
            p.show_plot("Potential Flow in a Room with Multiple Fans", "$x$ [m]", "$y$ [m]")


if __name__ == '__main__':
    sim = RoomSimulation(
        corners=np.array([
            [4, 4],
            [-4, 4],
            [-4, -4],
            [2, -4],
            [6, -2],
            [6, 2],
            [4, 4],
        ])
    )
    sim.plot_flowfield()
