import pandas as pd

class Params:

    def __init__(self, path, time_slow=0, dias=None):
        df_horizon = pd.read_excel(path, sheet_name="HORIZONTE")
        df_fleet = pd.read_excel(path, sheet_name="FROTA")
        df_grua = pd.read_excel(path, sheet_name="GRUA")
        df_bd_up = pd.read_excel(path, sheet_name="BD_UP")
        df_factory = pd.read_excel(path, sheet_name="FABRICA")
        df_routes = pd.read_excel(path, sheet_name="ROTA")
        # xls = pd.ExcelFile(path)
        # rota_sheets = [name for name in xls.sheet_names if name.startswith("ROTA")]
        # df_routes = pd.concat([pd.read_excel(xls, sheet_name=sheet) for sheet in rota_sheets], ignore_index=True)

        # ==================
        # DADOS DE ENTRADA
        # ==================

        # Conjuntos
        self.ups = list(df_bd_up.sort_values(by=['VOLUME'])['UP'].unique())
        # print({i: u for i, u in enumerate(self.ups)})
        self.farms = list(df_bd_up['FAZENDA'].unique())
        self.carries = list(df_fleet['TRANSPORTADOR'].unique())
        self.factories = list(df_factory['FABRICA'].unique())

        if dias is None:
            self.dias = df_horizon['DIA'].max()  # horizonte de planejamento (dias)
        else:
            self.dias = dias
        self.qt_i = len(self.ups)  # número de UPs
        self.qt_f = len(self.farms)  # número de fazendas
        self.qt_k = len(self.carries)  # número de transportadores
        self.qt_d = len(self.factories)  # número de fábricas

        self.I = range(self.qt_i)
        self.F = range(self.qt_f)
        self.K = range(self.qt_k)
        self.D = range(self.qt_d)
        self.T = range(self.dias)

        self.validate_data()

        # Parâmetros
        dict_db = dict(zip(df_bd_up['UP'], df_bd_up['DB']))
        self.DB = [dict_db[i] for i in self.ups]  # densidade básica da UP i

        dict_rsp = dict(zip(df_bd_up['UP'], df_bd_up['RSP']))
        self.RSP = [dict_rsp[i] for i in self.ups]  # relação sólido/polpa da UP i

        dem_min_dict = {
            (row['FABRICA'], row['DIA']): (row['DEMANDA_MIN'], row['DEMANDA_MAX'], row['RSP_MIN'], row['RSP_MAX'])
            for _, row in df_factory.iterrows()
        }
        self.DEM_min = [[dem_min_dict[(d, t + 1)][0] for t in self.T] for d in self.factories]
        self.DEM_max = [[dem_min_dict[(d, t + 1)][1] for t in self.T] for d in self.factories]

        self.RSP_min = [[dem_min_dict[(d, t + 1)][2] for t in self.T] for d in self.factories]
        self.RSP_max = [[dem_min_dict[(d, t + 1)][3] for t in self.T] for d in self.factories]

        dict_fmin = dict(zip(df_fleet['TRANSPORTADOR'], (df_fleet['FROTA_MIN'])))
        self.FROTA_min = [dict_fmin[k] for k in self.carries]

        dict_fmax = dict(zip(df_fleet['TRANSPORTADOR'], (df_fleet['FROTA_MAX'])))
        self.FROTA_max = [dict_fmax[k] for k in self.carries]

        dict_grua = dict(zip(df_grua['TRANSPORTADOR'], (df_grua['QTD_GRUAS'])))
        self.GRUA = [dict_grua[k] for k in self.carries]

        dict_cycle = dict(zip(df_routes['ORIGEM'], (df_routes['TEMPO_CICLO'])))
        cycle = [dict_cycle[i] for i in self.ups]

        dict_slowly = dict(zip(df_horizon['DIA'], (df_horizon['CICLO_LENTO'])))
        slowly = [int(dict_slowly[t + 1] == 'X') for t in self.T]

        dict_box = dict(zip(df_routes['DESTINO'], (df_routes['CAIXA_CARGA'])))
        box = [dict_box[d] for d in self.factories]

        self.CAPACIDADE = [[[box[d] * cycle[i] * (1 - time_slow * slowly[t]) for t in self.T] for d in self.D] for i in self.I]

        dict_teta = dict(zip(df_grua['TRANSPORTADOR'], (df_grua['PORCENTAGEM_VEICULOS_MIN'])))
        self.TETA = [dict_teta[k] for k in self.carries]

        dict_vi = dict(zip(df_bd_up['UP'], df_bd_up['VOLUME']))
        self.VI = [dict_vi[i] for i in self.ups]

        dict_p = dict(zip(df_bd_up['UP'], df_bd_up['FAZENDA']))
        self.P = [[int(dict_p[self.ups[i]] == self.farms[f]) for f in self.F] for i in self.I]
        self.M = df_fleet['FROTA_MAX'].sum()  # Big-M
        self.MV = df_bd_up['VOLUME'].sum()  # volume somado das UPs
        self.MD = df_bd_up['DB'].sum()
        self.L7 = [int(self.VI[i] < 7000) for i in self.I] # Não é m²
        self.U7 = [int(self.VI[i] >= 7000) for i in self.I]

        self.V_min = {
            (k, t): self.FROTA_min[k] * max(self.CAPACIDADE[i][d][t] for i in self.I for d in self.D)
            for k in self.K for t in self.T
        }

    def validate_data(self):
        pass
        # TODO: validar abas, cabeçalhos e tipos de dados das abas
        # TODO: filtrar UPs para usar apenas as que tem informações em todas as abas necessárias
        # TODO: filtrar transportadoras para usar apenas as necessárias
        # TODO: filtrar fábricas para usar apenas as necessárias
        # TODO: validar limites de RSP, demanda e frota
        # TODO: garantir que tabela de horizonte tenha informação para todos os dias
        # TODO: garantir que limite mínimo de veículos total não seja zero (ou alterar restrição do RSP ponderado)
        # TODO: garantir que CAPACIDADE não seja zero
        # TODO: garantir que não tenha UP sem volume inicial