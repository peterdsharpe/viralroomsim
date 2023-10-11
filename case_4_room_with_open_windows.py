import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from scipy import interpolate


class RoomSimulation:
    def __init__(self,
                 walls: list[np.ndarray],
                 ):
        points = []
        vortex_points = []
        collocation_points = []

        for wall in walls:
            upsample_resolution = 200 // len(wall)

            wall_points = interpolate.interp1d(
                np.linspace(0, 1, len(wall)),
                wall,
                axis=0,
            )(np.linspace(0, 1, upsample_resolution * (len(wall) - 1) + 1))

            # wall_vortex_points = wall_points[:-1, :]
            wall_collocation_points = (wall_points[:-1, :] + wall_points[1:, :]) / 2

            points.append(wall_points)
            vortex_points.append(wall_points)
            collocation_points.append(wall_collocation_points)

        points = np.concatenate(points)
        vortex_points = np.concatenate(vortex_points)
        collocation_points = np.concatenate(collocation_points)

        opti = asb.Opti()
        strengths = opti.variable(init_guess=np.zeros(len(vortex_points)))

        from potentialflowvisualizer import Flowfield, Doublet, Vortex

        fan = Doublet(
            strength=2,
            x=2,
            y=2,
            alpha=np.radians(40),  # Angle of the doublet axis
        )

        field = Flowfield(objects=[fan] + [
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
        normal_vectors /= np.linalg.norm(normal_vectors, axis=1, keepdims=True)

        wall_normal_velocity_at_collocation_points = np.sum(
            V_collocations * normal_vectors,
            axis=1
        )

        # opti.subject_to(np.sum(strengths) == 0)
        # opti.minimize(np.sum(wall_normal_velocity_at_collocation_points ** 2))
        opti.subject_to(np.sum(strengths) == 0)
        opti.subject_to(
            wall_normal_velocity_at_collocation_points == 0
        )
        opti.minimize(
            np.sum(strengths ** 2)
        )

        sol = opti.solve()

        self.walls = walls
        self.points = points
        self.vortex_points = vortex_points
        self.collocation_points = collocation_points
        self.normal_vectors = normal_vectors
        self.field: Flowfield = sol(field)
        self.fan = sol(fan)

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

    def plot_room(self, show=True, set_equal=True):
        for wall in self.walls:
            plt.plot(*wall.T, "-k", linewidth=3)
        if set_equal:
            p.equal()
        if show:
            p.show_plot(None, "$x$ [m]", "$y$ [m]")

    def plot_debug(self, show=True, set_equal=True):
        for wall in self.walls:
            plt.plot(*wall.T, "-k", alpha=0.3)
        plt.plot(*self.vortex_points.T, "o", alpha=0.4, markersize=3)
        plt.plot(*self.collocation_points.T, "o", alpha=0.4, markersize=3)
        plt.quiver(
            *self.collocation_points.T,
            *self.normal_vectors.T,
            color="r",
            scale=1,
            scale_units="xy",
            angles="xy",
        )
        if set_equal:
            p.equal()
        if show:
            p.show_plot(None, "$x$ [m]", "$y$ [m]")

    def plot_flowfield(self, show=True, set_equal=True):
        # xlim = np.array([self.points[:, 0].min(), self.points[:, 0].max()])
        # ylim = np.array([self.points[:, 1].min(), self.points[:, 1].max()])
        xlim = [-6, 8]
        ylim = [-6, 6]

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
        # cmap.set_under(plt.gca().get_facecolor())

        # Logarithmic normalization
        norm = LogNorm(vmin=1e-3, vmax=1)

        fig, ax = plt.subplots()

        X, Y = np.meshgrid(
            np.linspace(*xlim, 300),
            np.linspace(*ylim, 300),
        )
        U, V = self.get_velocity(X, Y)
        velmag = np.sqrt(U ** 2 + V ** 2)

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

        for wall in self.walls:
            plt.plot(*wall.T, "-k", linewidth=3)

        ax.add_patch(
            plt.Circle(
                xy=(self.fan.x, self.fan.y),
                radius=0.3,
                color="k",
                zorder=4,
            )
        )
        ax.add_patch(
            plt.arrow(
                x=self.fan.x - 0.2 * np.cos(self.fan.alpha),
                y=self.fan.y - 0.2 * np.sin(self.fan.alpha),
                dx=0.2 * np.cos(self.fan.alpha),
                dy=0.2 * np.sin(self.fan.alpha),
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
            p.show_plot("Potential Flow in a Room with Open Windows", "$x$ [m]", "$y$ [m]")


if __name__ == '__main__':
    sim = RoomSimulation(
        walls=[
            np.array([
                [4.5, 3.5],
                [4, 4],
                [-4, 4],
                [-4, 2],
            ]),
            np.array([
                [-4, 0],
                [-4, -4],
                [2, -4],
                [6, -2],
                [6, 2],
                [5.5, 2.5]
            ]),
        ]
    )
    sim.plot_flowfield(show=False)
    p.show_plot("Flow in a Room with Open Windows", "$x$ [m]", "$y$ [m]",
                savefig=["figures/case_4.svg",
                         "figures/case_4.png"]
                )
