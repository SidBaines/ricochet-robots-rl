import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from environment.utils import ROBOT_COLORS_PLOTLY, DIRECTION_NAMES

def action_to_human_readable(action):
    robot_idx = action // 4
    direction = DIRECTION_NAMES[action % 4]
    return f"{ROBOT_COLORS_PLOTLY[robot_idx]} Robot moving {direction}"

def plot_ricochet_robots_game(observations, values, actions, wall_thickness=6, grid_thickness=1):
    """
    observations: list of dicts, each with keys 'board_features' and 'target_robot_idx'
    values: list of floats (value estimates from agent)
    actions: list of ints (actions taken)
    robot_colors: list of color strings for robots
    """
    # --- Setup ---
    board_features = observations[0]['board_features']
    n_channels, height, width = board_features.shape
    num_robots = n_channels - 1 - 4  # [robots, target, 4 wall channels]
    wall_channels = [num_robots + 1 + i for i in range(4)]  # N, E, S, W

    # --- Helper: Draw a single frame ---
    def make_board_frame(obs):
        shapes = []
        # Draw grid lines (faint)
        for r in range(height + 1):
            shapes.append(dict(
                type="line",
                x0=0, y0=r, x1=width, y1=r,
                line=dict(color="lightgray", width=grid_thickness, dash="dot"),
                layer="below"
            ))
        for c in range(width + 1):
            shapes.append(dict(
                type="line",
                x0=c, y0=0, x1=c, y1=height,
                line=dict(color="lightgray", width=grid_thickness, dash="dot"),
                layer="below"
            ))

        # Draw walls (thick lines)
        for r in range(height):
            for c in range(width):
                # North wall
                if obs['board_features'][wall_channels[0], r, c]:
                    shapes.append(dict(
                        type="line",
                        x0=c, y0=r, x1=c+1, y1=r,
                        line=dict(color="black", width=wall_thickness),
                        layer="above"
                    ))
                # East wall
                if obs['board_features'][wall_channels[1], r, c]:
                    shapes.append(dict(
                        type="line",
                        x0=c+1, y0=r, x1=c+1, y1=r+1,
                        line=dict(color="black", width=wall_thickness),
                        layer="above"
                    ))
                # South wall
                if obs['board_features'][wall_channels[2], r, c]:
                    shapes.append(dict(
                        type="line",
                        x0=c, y0=r+1, x1=c+1, y1=r+1,
                        line=dict(color="black", width=wall_thickness),
                        layer="above"
                    ))
                # West wall
                if obs['board_features'][wall_channels[3], r, c]:
                    shapes.append(dict(
                        type="line",
                        x0=c, y0=r, x1=c, y1=r+1,
                        line=dict(color="black", width=wall_thickness),
                        layer="above"
                    ))

        # Draw robots (as colored circles)
        robot_x, robot_y, robot_marker, robot_color = [], [], [], []
        for i in range(num_robots):
            pos = np.argwhere(obs['board_features'][i] == 1)
            if len(pos) > 0:
                r, c = pos[0]
                robot_x.append(c + 0.5)
                robot_y.append(r + 0.5)
                robot_marker.append(i)
                robot_color.append(ROBOT_COLORS_PLOTLY[i % len(ROBOT_COLORS_PLOTLY)])

        # Draw target (as a star)
        target_pos = np.argwhere(obs['board_features'][num_robots] == 1)
        if len(target_pos) > 0:
            tr, tc = target_pos[0]
            target_x = tc + 0.5
            target_y = tr + 0.5
        else:
            target_x, target_y = None, None

        # Scatter for robots
        robot_scatter = go.Scatter(
            x=robot_x, y=robot_y,
            mode='markers+text',
            marker=dict(size=40, color=robot_color, line=dict(width=2, color='black')),
            text=[str(i) for i in robot_marker],
            textposition="middle center",
            name="Robots",
            showlegend=False
        )

        # Scatter for target
        target_scatter = go.Scatter(
            x=[target_x], y=[target_y],
            mode='markers',
            marker=dict(
                size=50, color=ROBOT_COLORS_PLOTLY[obs['target_robot_idx']], symbol='star', line=dict(width=2, color='black')
            ),
            name="Target",
            showlegend=False
        )

        return [robot_scatter, target_scatter], shapes

    # --- Build frames for animation ---
    plotly_frames = []
    for i, obs in enumerate(observations):
        data, shapes = make_board_frame(obs)
        frame = go.Frame(
            data=data,
            layout=go.Layout(
                shapes=shapes,
                title_text=f"Step {i+1}, Action: {action_to_human_readable(actions[i] if i < len(actions) else 'N/A')}"
            ),
            name=str(i)
        )
        plotly_frames.append(frame)

    # --- Initial plot setup ---
    # Value plot
    value_trace = go.Scatter(
        x=[0], y=[values[0]],
        mode='lines+markers',
        name='Value'
    )

    # Board plot (first frame)
    first_data, first_shapes = make_board_frame(observations[0])

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.5, 0.5],
        specs=[[{"type": "scatter"}, {"type": "scatter"}]]
    )
    for trace in first_data:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(value_trace, row=1, col=2)

    # Set axis for board
    fig.update_xaxes(range=[0, width], row=1, col=1, showgrid=False, zeroline=False, tickvals=list(range(width+1)))
    fig.update_yaxes(range=[height, 0], row=1, col=1, showgrid=False, zeroline=False, tickvals=list(range(height+1)))
    
    # Set axis for value plot
    fig.update_xaxes(range=[0, len(observations)-1], row=1, col=2, showgrid=True, zeroline=True)
    fig.update_yaxes(range=[min(values), max(values)], row=1, col=2, showgrid=True, zeroline=True)
    
    fig.update_layout(
        shapes=first_shapes,
        width=900, height=500,
        title="Ricochet Robots Agent Play",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        xaxis2=dict(title="Step"),
        yaxis2=dict(title="Value"),
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(label="⏮", method="animate", args=[["previous"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
                    dict(label="⏭", method="animate", args=[["next"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]),
                    dict(label="▶", method="animate", args=[None, {"frame": {"duration": 300, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
                    dict(label="⏸", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}])
                ],
                direction="left",
                pad={"r": 10, "t": 10},
                showactive=False,
                x=0.1,
                xanchor="right",
                y=1.1,
                yanchor="top"
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 20},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [i],
                        {"frame": {"duration": 0, "redraw": True},
                         "mode": "immediate",
                         "transition": {"duration": 0}}
                    ],
                    "label": str(i),
                    "method": "animate"
                } for i in range(len(observations))
            ]
        }]
    )

    # --- Add frames and value plot animation ---
    fig_frames = []
    for i, frame in enumerate(plotly_frames):
        # Value trace up to this step
        value_trace = go.Scatter(
            x=list(range(i+1)),
            y=values[:i+1],
            mode='lines+markers',
            name='Value',
            showlegend=False  # Hide legend for all but the first frame
        )
        # Combine the board data with the value trace
        frame.data = list(frame.data) + [value_trace]
        fig_frames.append(frame)
    fig.frames = fig_frames

    # Update the initial value trace to match the first frame
    fig.data[1].x = [0]
    fig.data[1].y = [values[0]]
    fig.data[1].showlegend = True  # Show legend only for the initial trace

    return fig