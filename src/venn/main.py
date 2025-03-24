import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random

# ============================
# Chargement de ton dataset
# ============================
df_ablation_all = pd.read_csv("all_models_ablation_per_dataset.csv")
df_ablation_all = df_ablation_all.round(2)

# ============================
# Paramètres
# ============================
radius = 4

center_classiques = np.array([0, 0])
center_spectral = np.array([4, 0])
center_topologie = np.array([2, 4])

categories_available = ['Classiques', 'Spectral', 'Topology']
model_to_keep = ['bert-base-uncased']

# ============================
# Couleurs et styles
# ============================
zone_styles = {
    'Classiques': ('blue', 'circle'),
    'Spectral': ('red', 'triangle-up'),
    'Topology': ('green', 'square'),
    'Classiques, Spectral': ('purple', 'star'),
    'Classiques, Topology': ('orange', 'cross'),
    'Spectral, Topology': ('brown', 'x'),
    'Classiques, Spectral, Topology': ('black', 'diamond')
}

pastel_colors = {
    'Classiques': 'rgba(173, 216, 230, 0.3)',  # ✅ Jaune pastel
    'Spectral': 'rgba(255, 192, 203, 0.3)',    # Rose pastel clair
    'Topology': 'rgba(144, 238, 144, 0.3)'     # Vert pastel clair
}
# Mapping plotly marker ➡️ unicode symbol
marker_symbols_unicode = {
    'circle': '●',
    'triangle-up': '▲',
    'square': '■',
    'diamond': '◆',
    'cross': '✚',
    'x': '✖',
    'star': '★'
}


# ============================
# Fonctions pour ton ablation
# ============================
def get_results(df_ablation_all, model_to_keep, categories):
    categories_to_avoid = set(['Classiques', 'Topology', 'Spectral']) - set(categories)
    df_ablation_filtered = df_ablation_all[~df_ablation_all['Categories_Distances'].apply(
        lambda x: any(categorie in x.split(', ') for categorie in categories_to_avoid))]

    df_ablation_filtered = df_ablation_filtered[df_ablation_filtered.Model.isin(model_to_keep)]

    category_dist_dict = {}
    for i, dataset in enumerate(df_ablation_filtered.Dataset.unique()):
        max_val = df_ablation_filtered[df_ablation_filtered.Dataset == dataset].F1.max()
        df_petit = df_ablation_filtered[(df_ablation_filtered.Dataset == dataset) & (df_ablation_filtered.F1 == max_val)]
        for category_dist in df_petit.Categories_Distances:
            if category_dist in category_dist_dict:
                category_dist_dict[category_dist].append(i)
            else:
                category_dist_dict[category_dist] = [i]

    return {i: set(j) for i, j in category_dist_dict.items()}

# ============================
# Fonctions géométriques
# ============================
def in_circle(p, center, radius):
    return np.linalg.norm(p - center) <= radius

def condition_only(center_in, centers_out, radius):
    def func(p):
        if not in_circle(p, center_in, radius):
            return False
        for c_out in centers_out:
            if in_circle(p, c_out, radius):
                return False
        return True
    return func

def condition_intersection(centers_in, centers_out, radius):
    def func(p):
        for c_in in centers_in:
            if not in_circle(p, c_in, radius):
                return False
        for c_out in centers_out:
            if in_circle(p, c_out, radius):
                return False
        return True
    return func

def condition_intersection_3_cercles(center1, center2, center3, radius):
    def func(p):
        return (
            in_circle(p, center1, radius) and
            in_circle(p, center2, radius) and
            in_circle(p, center3, radius)
        )
    return func

def generate_points(condition_func, center_ref, radius, n_points, max_trials=10000):
    points = []
    trials = 0
    while len(points) < n_points:
        if trials > max_trials:
            break
        r = radius * np.sqrt(random.uniform(0, 1)) * 0.7
        theta = random.uniform(0, 2 * np.pi)
        x = center_ref[0] + r * np.cos(theta)
        y = center_ref[1] + r * np.sin(theta)
        p = np.array([x, y])
        if condition_func(p):
            points.append((x, y))
        trials += 1
    return points

# ============================
# Dash App Layout
# ============================
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Diagramme de Venn dynamique avec ablation", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Choisissez les catégories à afficher :", style={'fontWeight': 'bold'}),
        dcc.Checklist(
            id='categories-checklist',
            options=[{'label': cat, 'value': cat} for cat in categories_available],
            value=categories_available,
            inline=True
        )
    ], style={'marginBottom': '20px'}),

    dcc.Store(id='store-assignments', data={}),

    dcc.Graph(id='venn-graph'),

    html.Div(id='datasets-deplaces', style={'marginTop': '20px'})
])

# ============================
# Callback ➡️ Mise à jour du graph + détecter les déplacements
# ============================
@app.callback(
    [Output('venn-graph', 'figure'),
     Output('store-assignments', 'data'),
     Output('datasets-deplaces', 'children')],
    [Input('categories-checklist', 'value')],
    [State('store-assignments', 'data')]
)
def update_graph(selected_categories, previous_assignments):
    # 🔸 Résultats après ablation
    results = get_results(df_ablation_all, model_to_keep, selected_categories)

    # 🔸 Définir les centres des cercles affichés
    centers = {}
    if 'Classiques' in selected_categories:
        centers['Classiques'] = center_classiques
    if 'Spectral' in selected_categories:
        centers['Spectral'] = center_spectral
    if 'Topology' in selected_categories:
        centers['Topology'] = center_topologie

    fig = go.Figure()

    # =======================
    # Cercles + Labels
    # =======================
    for i, (label, center) in enumerate(centers.items()):
        pastel_color = pastel_colors[label]
        border_color = pastel_color.replace('0.3', '1')

        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=center[0] - radius, y0=center[1] - radius,
            x1=center[0] + radius, y1=center[1] + radius,
            line=dict(
                color=border_color,
                width=3
            ),
            fillcolor=pastel_color,
            opacity=0.6,
            layer='below'
        )

        fig.add_annotation(
            x=-5,
            y=5 + i,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=16, color='black', family="Arial Black"),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=border_color,
            borderwidth=2,
            borderpad=4
        )

    # =======================
    # Générer les points et stocker l'affectation actuelle
    # =======================
    current_assignments = {}
    for zone_name, dataset_ids in results.items():
        dataset_ids = sorted(dataset_ids)
        if len(dataset_ids) == 0:
            continue

        zone_list = zone_name.split(', ')
        in_centers = [centers[cat] for cat in zone_list if cat in centers]
        out_centers = [c for cat, c in centers.items() if cat not in zone_list]

        if len(in_centers) == 1:
            condition = condition_only(in_centers[0], out_centers, radius)
            center_ref = in_centers[0]
        elif len(in_centers) == 2:
            condition = condition_intersection(in_centers, out_centers, radius)
            center_ref = np.mean(in_centers, axis=0)
        elif len(in_centers) == 3:
            condition = condition_intersection_3_cercles(*in_centers, radius)
            center_ref = np.mean(in_centers, axis=0)
        else:
            continue

        points = generate_points(condition, center_ref, radius, len(dataset_ids))

        color, marker = zone_styles.get(zone_name, ('grey', 'circle'))

        for dataset_id, (x, y) in zip(dataset_ids, points):
            dataset_id_str = str(dataset_id)
            current_assignments[dataset_id_str] = zone_name

            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                name=f"Dataset {dataset_id}",
                marker=dict(
                    color=color,
                    symbol=marker,
                    size=12,
                    line=dict(width=1, color='black')
                ),
                hovertemplate=f'Dataset ID: {dataset_id}',
                showlegend=True
            ))

        # 💡 Labels avec le nombre d'éléments
        x_shift = np.array([0, 0])
        if len(selected_categories) == 3:
            if zone_name == 'Classiques':
                x_shift += np.array([-1, 0])
            elif zone_name == 'Spectral':
                x_shift += np.array([1, 0])
            elif zone_name == 'Topology':
                x_shift += np.array([0, 1])
            elif zone_name == 'Classiques, Spectral':
                x_shift += np.array([0, -1])
            elif zone_name == 'Classiques, Topology':
                x_shift += np.array([-1, 1])
            elif zone_name == 'Spectral, Topology':
                x_shift += np.array([1, 1])
        elif len(selected_categories) == 2:
            if zone_name == 'Classiques':
                x_shift += np.array([-1, 0])
            elif zone_name == 'Spectral':
                x_shift += np.array([1, 0])
            elif zone_name == 'Topology':
                x_shift += np.array([0, 1])

        fig.add_annotation(
            x=center_ref[0] + x_shift[0],
            y=center_ref[1] + x_shift[1],
            text=str(len(dataset_ids)),
            showarrow=False,
            font=dict(size=26, color='black', family="Arial"),
            bgcolor="rgba(255,255,255,1)"
        )

    # ============================
    # Identifier les changements
    # ============================
    changements = []
    previous_assignments={'0': 'Classiques, Spectral, Topology', '1': 'Classiques, Spectral, Topology', '7': 'Classiques, Topology', '10': 'Classiques, Spectral, Topology', '14': 'Classiques, Spectral, Topology', '17': 'Classiques, Spectral, Topology', '20': 'Classiques, Spectral, Topology', '22': 'Classiques, Spectral, Topology', '25': 'Classiques, Spectral, Topology', '26': 'Classiques, Spectral', '36': 'Classiques, Spectral, Topology', '40': 'Classiques, Spectral, Topology', '42': 'Classiques', '53': 'Classiques, Spectral, Topology', '56': 'Classiques, Spectral, Topology', '58': 'Classiques, Spectral, Topology', '59': 'Classiques, Spectral, Topology', '62': 'Classiques, Spectral, Topology', '68': 'Classiques, Spectral, Topology', '71': 'Classiques, Spectral, Topology', '73': 'Classiques, Spectral, Topology', '74': 'Classiques, Spectral, Topology', '75': 'Classiques, Spectral, Topology', '76': 'Classiques, Spectral, Topology', '81': 'Classiques, Spectral, Topology', '84': 'Classiques, Spectral, Topology', '87': 'Classiques, Spectral, Topology', '88': 'Classiques, Spectral, Topology', '89': 'Classiques, Spectral, Topology', '90': 'Classiques, Spectral, Topology', '91': 'Classiques, Spectral, Topology', '97': 'Classiques, Spectral, Topology', '99': 'Classiques, Spectral, Topology', '100': 'Classiques, Spectral, Topology', '101': 'Classiques, Spectral, Topology', '102': 'Classiques, Spectral, Topology', '105': 'Classiques, Spectral, Topology', '107': 'Classiques, Spectral, Topology', '108': 'Classiques, Spectral, Topology', '110': 'Classiques, Spectral, Topology', '2': 'Classiques, Spectral, Topology', '57': 'Classiques, Spectral, Topology', '106': 'Classiques, Spectral, Topology', '3': 'Classiques, Spectral, Topology', '5': 'Classiques, Spectral, Topology', '6': 'Classiques, Topology', '8': 'Classiques, Spectral, Topology', '11': 'Classiques, Spectral, Topology', '12': 'Classiques, Spectral, Topology', '13': 'Classiques, Spectral, Topology', '15': 'Classiques, Topology', '16': 'Classiques, Spectral, Topology', '18': 'Classiques, Spectral, Topology', '19': 'Classiques, Spectral, Topology', '21': 'Classiques, Topology', '23': 'Classiques, Spectral, Topology', '24': 'Classiques, Spectral, Topology', '27': 'Classiques, Spectral, Topology', '28': 'Classiques, Spectral, Topology', '29': 'Classiques, Topology', '31': 'Classiques, Spectral, Topology', '32': 'Classiques, Topology', '34': 'Classiques, Spectral, Topology', '35': 'Classiques, Spectral, Topology', '37': 'Classiques, Spectral, Topology', '39': 'Classiques, Spectral, Topology', '41': 'Classiques, Spectral, Topology', '43': 'Classiques, Spectral, Topology', '45': 'Classiques, Topology', '47': 'Classiques, Spectral, Topology', '48': 'Classiques, Spectral, Topology', '49': 'Classiques, Topology', '52': 'Classiques, Spectral, Topology', '54': 'Classiques, Spectral, Topology', '55': 'Classiques, Spectral, Topology', '60': 'Classiques, Topology', '61': 'Classiques, Spectral, Topology', '63': 'Classiques, Spectral, Topology', '64': 'Classiques, Spectral, Topology', '65': 'Classiques, Spectral, Topology', '66': 'Classiques, Spectral, Topology', '67': 'Classiques, Topology', '69': 'Classiques, Spectral, Topology', '70': 'Classiques, Spectral, Topology', '72': 'Classiques, Spectral, Topology', '77': 'Classiques, Spectral, Topology', '78': 'Classiques, Spectral, Topology', '79': 'Classiques, Spectral, Topology', '80': 'Classiques, Spectral, Topology', '82': 'Classiques, Topology', '83': 'Classiques, Spectral, Topology', '85': 'Classiques, Spectral, Topology', '86': 'Classiques, Spectral, Topology', '92': 'Classiques, Spectral, Topology', '93': 'Classiques, Spectral, Topology', '95': 'Classiques, Topology', '96': 'Classiques, Spectral, Topology', '98': 'Classiques, Spectral, Topology', '103': 'Classiques, Spectral, Topology', '104': 'Classiques, Spectral, Topology', '109': 'Classiques, Spectral, Topology', '4': 'Classiques, Spectral, Topology', '9': 'Classiques, Spectral, Topology', '30': 'Classiques, Spectral, Topology', '33': 'Classiques, Spectral, Topology', '38': 'Classiques, Spectral, Topology', '44': 'Classiques, Spectral, Topology', '46': 'Classiques, Spectral, Topology', '50': 'Classiques, Spectral, Topology', '51': 'Classiques, Spectral, Topology', '94': 'Classiques, Spectral, Topology', '111': 'Classiques, Spectral, Topology'}

    for dataset_id_str, new_zone in current_assignments.items():
        old_zone = previous_assignments.get(dataset_id_str, "")

        if old_zone != new_zone and  old_zone != "" :
            # 🔸 Anciennes couleurs/formes
            old_color, old_marker = zone_styles.get(old_zone, ('grey', 'circle'))
            # 🔸 Nouvelles couleurs/formes
            new_color, new_marker = zone_styles.get(new_zone, ('grey', 'circle'))

            changements.append(
                html.Li([
                    html.Span(f"Dataset {dataset_id_str} : ", style={"font-weight": "bold"}),

                    html.Span([
                        marker_symbols_unicode.get(old_marker, '?'),
                        f" {old_zone}"
                    ], style={
                        "color": old_color,
                        "border": f"2px solid {old_color}",
                        "padding": "2px 5px",
                        "margin-right": "5px",
                        "background-color": "white",
                        "border-radius": "4px",
                        "font-size": "18px"  # Tu peux ajuster la taille ici
                    }),

                    html.Span(" ➡️ "),

                    html.Span([
                        marker_symbols_unicode.get(new_marker, '?'),
                        f" {new_zone}"
                    ], style={
                        "color": new_color,
                        "border": f"2px solid {new_color}",
                        "padding": "2px 5px",
                        "margin-left": "5px",
                        "background-color": "white",
                        "border-radius": "4px",
                        "font-size": "18px"
                    })
                ])
            )




    if changements:
        changements_html = html.Div([
            html.H4("Datasets déplacés après ablation :", style={'color': 'red'}),
            html.Ul(changements)
        ])
    else:
        changements_html = html.Div("Aucun dataset déplacé.", style={'color': 'green'})

    # ============================
    # Layout final
    # ============================
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(
            title="Datasets",
            orientation="v",
            x=1.02,
            y=1
        )
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig, current_assignments, changements_html

# ============================
# RUN SERVER
# ============================
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))  # Render provides a PORT env variable
    app.run(host="0.0.0.0", port=port, debug=True)

fig.write_html("venn_diagram.html", full_html=True, include_plotlyjs='cdn')

