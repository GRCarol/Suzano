import webbrowser
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import os

def plot_gantt(data, x, v, init=datetime(year=2025, month=1, day=1)):
    # === 1) Monte um DataFrame com o resultado ===

    # EXEMPLO: Supondo que você já tenha a solução salva como um dict
    # (ou pegue diretamente de x[i,d,k,t].X do Gurobi)

    # Vamos criar uma lista de registros
    html = ''
    rows = []
    for i in data.I:
        fazenda = next(farm for f, farm in enumerate(data.farms) if data.P[i][f] == 1) # ajuste para seu atributo de fazenda por UP
        for d in data.D:
            for t in data.T:
                for k in data.K:
                    # Soma volume de todos transportadores
                    vol = x[i,d,k,t].X * data.CAPACIDADE[i][d][t]
                    if vol > 0:
                        rows.append({
                            "UP": i,
                            "Fazenda": fazenda,
                            "Fabrica": d,
                            "DiaInicio": init + timedelta(days=t),
                            "DiaFim": (init + timedelta(days=t)).replace(hour=23, minute=59, second=59),
                            "Volume": vol,
                            "Transportadora": data.carries[k],
                            "Caminhões enviados": x[i,d,k,t].X,
                            "Capacidade": data.CAPACIDADE[i][d][t],
                        })

    # Converta em DataFrame
    df = pd.DataFrame(rows)

    # Para garantir que UPs fiquem ordenadas por Fazenda
    df['Fazenda_UP'] = df['Fazenda'].astype(str) + "_" + df['UP'].astype(str)
    df = assign_gantt_slots(df)
    df['Fazenda_UP_Slot'] = df['Fazenda_UP'] + "_Slot" + df['Slot'].astype(str)

    # === 2) Plote um Gantt por fábrica ===

    for fabrica in df['Fabrica'].unique():
        df_fab = df[df['Fabrica'] == fabrica].copy()

        # Ordenar UPs por fazenda para eixo y
        ordenadas = df_fab[['Fazenda', 'UP']].drop_duplicates()
        ordenadas = ordenadas.sort_values(['Fazenda', 'UP'])
        up_order = [f"{row['Fazenda']}_{row['UP']}" for idx, row in ordenadas.iterrows()]
        df_fab['Fazenda_UP'] = df_fab['Fazenda'].astype(str) + "_" + df_fab['UP'].astype(str)
        df_fab['DiaInicio'] = pd.to_datetime(df_fab['DiaInicio'])
        df_fab['DiaFim'] = pd.to_datetime(df_fab['DiaFim'])
        df_fab['UP_code'] = df_fab['UP'].apply(lambda a: data.ups[a])

        fig = px.timeline(
            df_fab,
            x_start="DiaInicio",
            x_end="DiaFim",
            y="Fazenda_UP_Slot",
            color="Transportadora",
            text="Volume",
            category_orders={"Fazenda_UP_Slot": sorted(df_fab['Fazenda_UP_Slot'].unique())},
            title=f"Gantt de Transporte - Fábrica {fabrica}",
            hover_data=['Capacidade', 'Caminhões enviados', 'UP_code']
        )

        fig.update_yaxes(
            autorange="reversed",
            title="Fazenda - UP"
        )
        fig.update_xaxes(
            title="Dia"
        )
        fig.update_traces(
            marker=dict(line=dict(width=0.5, color="black")),
            texttemplate="%{text:.1f}",
            textposition="inside"
        )

        # fig.show()
        html += f"gantt_fabrica_{fabrica}.html"
        html += fig.to_html()
        # fig.write_html(html_name)
        # webbrowser.open_new_tab(html_name)
    return html

def volume_by_up(data, x, v, init=datetime(year=2025, month=1, day=1)):
    # === 1) Volume por UP ===
    html = ''
    all_up = []
    for f in data.F:
        fazenda = data.farms[f]
        rows_up = []
        for i in data.I:
            if data.P[i][f] == 0:
                continue

            up_info = {
                "UP": i,
                "Fazenda": fazenda,
                "Dia": init,
                "Volume": data.VI[i]
            }
            rows_up.append(up_info)
            all_up.append(up_info)
            for t in data.T:
                up_info = {
                    "UP": i,
                    "Fazenda": fazenda,
                    "Dia": init + timedelta(days=t + 1),
                    "Volume": v[i, t].X
                }
                rows_up.append(up_info)
                all_up.append(up_info)

        df_up = pd.DataFrame(rows_up)

        fig1 = px.line(df_up, x="Dia", y="Volume", color="UP",
                       title=f"Volume por UP por dia da fazenda {fazenda}")
        html += fig1.to_html()
    df_all = pd.DataFrame(all_up)

    return html, df_all

def volume_by_farm(df_up):
    # === 2) Volume por Fazenda ===

    # Basta agrupar o df_up:
    df_fazenda = df_up.groupby(['Fazenda', 'Dia']).agg({"Volume": "sum"}).reset_index()

    fig2 = px.line(df_fazenda, x="Dia", y="Volume", color="Fazenda",
                   title="Volume transportado por Fazenda por dia")
    # fig2.show()
    return fig2.to_html()

def show_trucks(data, x, init=datetime(year=2025, month=1, day=1)):
    # === 3) Viagens por Transportadora ===

    rows_k = []
    for k in data.K:
        for t in data.T:
            distribution = {i: x[i, d, k, t].X for i in data.I for d in data.D}
            distribution = {a: b for a, b in distribution.items() if b > 0}
            viagens = sum(distribution.values())
            percentual = {a: round(b/viagens, 2) for a, b in distribution.items()}
            rows_k.append({
                "Transportadora": data.carries[k],
                "Dia": init + timedelta(days=t),
                "Viagens": viagens,
                "Distribuição": str(distribution),
                "Percentual": str(percentual)
            })

    df_k = pd.DataFrame(rows_k)

    fig3 = px.line(df_k, x="Dia", y="Viagens", color="Transportadora", hover_data=['Distribuição', 'Percentual'],
                   title="Número de caminhões por Transportadora por dia")
    # fig3.show()
    return fig3.to_html()

def assign_gantt_slots(df):
    """
    Recebe um df com colunas:
      - UP (ou Fazenda_UP)
      - DiaInicio
      - DiaFim

    Retorna o mesmo df com uma coluna "Slot" para evitar sobreposição.
    """
    out_rows = []
    for up, subdf in df.groupby("Fazenda_UP"):
        # Ordene barras por início
        events = subdf.sort_values("DiaInicio").copy()
        slots = []  # Lista de slots ativos: (end_time, slot_id)
        slots_per_row = []

        for idx, row in events.iterrows():
            # Tente encaixar em um slot livre
            assigned = False
            for s_idx, (slot_end, slot_id) in enumerate(slots):
                if row['DiaInicio'] > slot_end:
                    # Sem sobreposição, pode usar este slot
                    slots[s_idx] = (row['DiaFim'], slot_id)
                    slots_per_row.append(slot_id)
                    assigned = True
                    break
            if not assigned:
                # Crie novo slot
                new_slot = len(slots) + 1
                slots.append((row['DiaFim'], new_slot))
                slots_per_row.append(new_slot)

        events['Slot'] = slots_per_row
        out_rows.append(events)

    return pd.concat(out_rows)


def build_html(data, x, v, init=datetime(year=2025, month=1, day=1)):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pasta_saida = os.path.join(base_dir, "output")
    os.makedirs(pasta_saida, exist_ok=True)
    html = plot_gantt(data, x, v, init)
    html_, df_up = volume_by_up(data, x, v, init)
    html += html_
    html += volume_by_farm(df_up)
    html += show_trucks(data, x, init)
    file_path = os.path.join(pasta_saida, "result.html")
    with open(file_path, "w", encoding="utf-8") as html_file:   
        html_file.write(html)
    webbrowser.open_new_tab(file_path)

