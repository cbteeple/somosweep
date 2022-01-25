import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import numpy as np
from copy import deepcopy


class Section:
    def __init__(
        self, title, length, num_rows, num_cols, fig, buffer_len, component_names=None
    ):
        self.title = title
        self.length = length
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.artists = [None] * length
        self.ylim = (-5, 5)  # TODO: Better way of setting this
        self.fig = fig
        self.buffer_len = buffer_len
        self.component_names = (
            range(length) if component_names is None else component_names
        )
        self.state_buffers = [[0] * buffer_len for _ in range(length)]

    def setup_plot(self, section_num, subplot_spec, step_nums):
        gs_sec_head = mpl.gridspec.GridSpecFromSubplotSpec(
            1, 1, subplot_spec=subplot_spec[2 + (section_num * 3)]
        )
        heading_ax = self.fig.add_subplot(gs_sec_head[0])
        heading_ax.set_title(self.title, fontsize=16)
        heading_ax.set_axis_off()
        gs_sec_plots = mpl.gridspec.GridSpecFromSubplotSpec(
            self.num_rows,
            self.num_cols,
            wspace=0.3,
            subplot_spec=subplot_spec[3 + (section_num * 3)],
        )

        count = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if count != self.length:
                    plot_ax = self.fig.add_subplot(gs_sec_plots[i, j])
                    plot_ax.set_ylim(*self.ylim)
                    self.artists[count] = plot_ax.plot(
                        step_nums, [0] * len(step_nums), animated=True
                    )[0]
                    plot_ax.annotate(
                        str(self.component_names[count]),
                        (0, 1),
                        xycoords="axes fraction",
                        xytext=(2, -2),
                        textcoords="offset points",
                        ha="left",
                        va="top",
                        fontweight="bold",
                    )
                    count += 1

    def update_artists(self, new_vals):
        # new vals is in form [x0, x1, x2, ...]

        for i, artist in enumerate(self.artists):
            self.state_buffers[i].append(new_vals[i])
            self.state_buffers[i] = self.state_buffers[i][-self.buffer_len :]
            artist.set_ydata(self.state_buffers[i])


class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for sub-classes of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)
        # let the GUI event loop process anything it has to do
        cv.flush_events()


class Debugger:  # TODO: good way of selecting debug section from training/env creation.
    def __init__(self, env, sections=None):
        self.env = env
        self.run_ID = env.run_ID
        all_sections = [
            "reward_components",
            "observations",
            "actions",
            "applied_torques",
        ]
        self.selected_sections = (
            all_sections if sections in [None, True, False] else sections
        )
        self.buffer_len = 100

    # called on first env reset
    def setup(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.prev_total_rewards = deepcopy(self.env.reset_reward_component_info)
        section_lens = {
            "reward_components": len(self.prev_total_rewards),
            "observations": self.env.observation_space.shape[0],
            "actions": self.env.action_space.shape[0],
            "applied_torques": self.env.action_space.shape[0],
        }
        self.sections = [None] * len(self.selected_sections)
        height_ratios = [200]
        max_num_cols = 6

        for i, section_name in enumerate(self.selected_sections):
            num_rows = int(np.ceil(section_lens[section_name] / max_num_cols))
            num_cols = int(min(section_lens[section_name], max_num_cols))
            component_names = None
            if section_name == "reward_components":
                component_names = list(self.prev_total_rewards.keys())
            section = Section(
                section_name,
                section_lens[section_name],
                num_rows,
                num_cols,
                self.fig,
                self.buffer_len,
                component_names,
            )
            self.sections[i] = section
            height_ratios += [1, 0.2, 100 * num_rows]

        # building layers (without data)
        gs_master = mpl.gridspec.GridSpec(
            len(height_ratios), 1, height_ratios=height_ratios, hspace=0, wspace=0
        )
        self.fig.suptitle("SoMo-RL Debug Dashboard", fontsize=24)

        ### ----------------------------- ## -----------------------------
        ## layer 1 - General Data
        # left half
        self.info_text = [None] * 3
        gs_2 = mpl.gridspec.GridSpecFromSubplotSpec(
            1, 2, height_ratios=[1], subplot_spec=gs_master[0]
        )
        info_text_ax = self.fig.add_subplot(gs_2[0])

        # info text panel
        info_text_ax.text(
            0.03, 0.85, "Run Information:", fontsize=11, fontweight="bold"
        )
        info_text_ax.text(0.03, 0.70, f"Run ID: {self.run_ID}", fontsize=11)
        # todo: update this in main branch; handle better

        try:
            max_episode_steps = self.env.run_config["max_episode_steps"]
        except:
            max_episode_steps = self.env.spec.max_episode_steps

        info_text_ax.text(
            0.03,
            0.10,
            f"Max Steps/Episode: {max_episode_steps}",
            fontsize=11,
        )
        self.info_text[0] = info_text_ax.text(
            0.03, 0.55, "Time Elapsed: 0", fontsize=11, animated=True
        )
        self.info_text[1] = info_text_ax.text(
            0.03, 0.40, "Current Episode: 0", fontsize=11, animated=True
        )
        self.info_text[2] = info_text_ax.text(
            0.03, 0.25, "Current Step: 0", fontsize=11, animated=True
        )
        info_text_ax.axes.xaxis.set_visible(False)
        info_text_ax.axes.yaxis.set_visible(False)

        # -------------------------------
        # right half
        self.reward = [None] * 1

        reward_ax = self.fig.add_subplot(gs_2[1])

        # reward/step panel
        reward_ax.set_title("Reward History")
        reward_ax.set_ylabel("Reward Value")
        reward_ax.set_xlabel("Number Steps in Past")
        self.r_per_step = [0] * self.buffer_len
        step_nums = range(-self.buffer_len + 1, 1)
        step_r_ylim = (-300, 200)  # TODO: Better way of setting this
        reward_ax.set_ylim(*step_r_ylim)
        self.reward[0] = reward_ax.plot(step_nums, self.r_per_step, animated=True)[0]
        reward_ax.grid()

        ### ----------------------------- ## -----------------------------
        ## layer 2 - Selected Sections
        sections_artists = []
        for i, section in enumerate(self.sections):
            section.setup_plot(i, gs_master, step_nums)
            sections_artists += section.artists

        ## ----------------------------- ## -----------------------------
        # joins layers
        gs_master.tight_layout(self.fig, pad=0.5, h_pad=-1)

        # set up blitter
        self.bm = BlitManager(
            self.fig.canvas,
            [
                *self.info_text,
                *self.reward,
                *sections_artists,
            ],
        )
        plt.show(block=False)
        plt.pause(0.1)

        self.episode = 0
        self.start_t = time.time()

    def update(self, reward, action, observation):
        time_elapsed = time.time() - self.start_t
        step = self.env.step_count
        ep = self.env.ep_count
        reward_components_totals = deepcopy(self.env.reward_component_info)
        step_rewards = [0] * len(self.prev_total_rewards)
        if "reward_components" in self.selected_sections:
            for i, key in enumerate(self.prev_total_rewards):
                step_rewards[i] = (
                    reward_components_totals[key] - self.prev_total_rewards[key]
                )
            self.prev_total_rewards = deepcopy(reward_components_totals)

        updated_state = {
            "reward_components": step_rewards,
            "observations": observation,
            "actions": action,
            "applied_torques": self.env.applied_torque,
        }

        self.r_per_step.append(reward)
        self.r_per_step = self.r_per_step[-self.buffer_len :]

        # info panel
        self.info_text[0].set_text(
            "Time Elapsed (sec): " + str(np.round(time_elapsed, 0))
        )
        self.info_text[1].set_text("Current Episode: " + str(ep))
        self.info_text[2].set_text("Current Step: " + str(step))

        # reward/step panel
        self.reward[0].set_ydata(self.r_per_step)

        # sections
        for section in self.sections:
            section.update_artists(updated_state[section.title])

        self.bm.update()
