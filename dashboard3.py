import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import datetime
# import seaborn as sns
import plotly.express as px
# import missingno as msn
from PIL import Image

año_actual = datetime.datetime.now().year

ruta_logo = "logo3_uta.png"
logo = Image.open(ruta_logo)
tamaño_logo = (290,290)

usuarios_permitidos = {
    "usuario": "dsl2024"
}

st.set_page_config(layout="wide", page_title="Dashboard DSL", page_icon=":chart_with_upwards_trend:", initial_sidebar_state="collapsed")
st.sidebar.image(logo,width=tamaño_logo[0])

#@st.cache_data (experimental_allow_widgets=True)

def mostrar_login():
    st.title("Login")
    username = st.text_input("Nombre de usuario")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar sesión"):
        if username in usuarios_permitidos and usuarios_permitidos[username] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.rerun()
        else:
            st.error("Nombre de usuario o contraseña incorrectos")

def cargar_datos(archivo_excel):
    return pd.read_excel(archivo_excel)

def df_nulos(df): 
    pass
    # fig, ax = plt.subplots()
    # msn.matrix(df,ax=ax, fontsize=5.5,sparkline=False)
    # return st.pyplot(fig)

def obtener_color(porc_pend):
    if porc_pend >= 50:
        return 'color: red; font-size: 50px; text-align: center'
    elif 31 <= porc_pend <= 49:
        return 'color: orange; font-size: 50px; text-align: center'
    elif 0 <= porc_pend <= 30:
        return 'color: #5DD39E; font-size: 50px; text-align: center'

def porc_pend_dashboard_anual(df, año):
    ejec = df.Ejecutada[(df.Año == año) & (df.Ejecutada == "Si")].value_counts()
    pend = df.Ejecutada[(df.Año == año) & (df.Ejecutada == "No")].value_counts()
    total = np.sum(ejec) + np.sum(pend)
    if total > 0:
        porc = round(np.sum(pend) / total * 100)
    else:
        porc = 0

    color = obtener_color(porc)

    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Porcentaje de OT pendientes"

    metric_html = f"<div style= '{div_style}'>"\
                  f"<span style= '{title_style}'>{titulo}</span></br>"\
                  f"<span style= '{color}'>{porc}%</span></div>"
    if total == 0:
        return st.write("Periodo no registrado")
    else:
        return st.write(metric_html, unsafe_allow_html=True)

def porc_ejec_dashboard_anual(df, año):
    ejec = df.Ejecutada[(df.Año == año) & (df.Ejecutada == "Si")].value_counts()
    pend = df.Ejecutada[(df.Año == año) & (df.Ejecutada == "No")].value_counts()
    total = np.sum(ejec) + np.sum(pend)

    if total > 0:  # Asegurarse de que total no sea cero para evitar división por cero
        porc = round(np.sum(ejec) / total * 100)
    else:
        porc = 0

    porc_pend = 100 - porc  # Calcular el porcentaje de pendientes
    color = obtener_color(porc_pend)

    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Porcentaje de OT ejecutadas"

    metric_html = f"<div style= '{div_style}'>"\
                  f"<span style= '{title_style}'>{titulo}</span></br>"\
                  f"<span style= '{color}'>{porc}%</span></div>"
    if total == 0:
        return st.write("Periodo no registrado")
    else:
        return st.write(metric_html, unsafe_allow_html=True)
    
def cant_pend_dashboard_anual(df, año):
    pend = df.Ejecutada[(df.Ejecutada=="No")&(df.Año==año)].count()

    color = 'color: #808B96; font-size: 50px; text-align: center'
    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Cantidad de pendientes"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{pend}</span></div>"
    
    return st.write(metric_html,unsafe_allow_html=True)

def cant_ejec_dashboard_anual(df, año):    
    ejec = df.Ejecutada[(df.Ejecutada=="Si")&(df.Año==año)].count()

    color = 'color: #808B96; font-size: 50px; text-align: center'
    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Cantidad de ejecutadas"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{ejec}</span></div>"
    
    return st.write(metric_html,unsafe_allow_html=True)

def porc_ejec_dashboard_servicio(df):
    ejec = df.Ejecutada[df.Ejecutada=="Si"].value_counts()
    pend = df.Ejecutada[df.Ejecutada=="No"].value_counts()
    total = np.sum(ejec)+np.sum(pend)
    porc = round(np.sum(ejec)/total*100)

    porc_ejec = round(np.sum(pend)/total*100)

    color = obtener_color(porc_ejec)

    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Porcentaje de OT ejecutadas"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{porc}%</span></div>"
    if total == 0:
        return st.write("Periodo no registrado")
    else:
        return st.write(metric_html,unsafe_allow_html=True)

def porc_no_ejec_dashboard_servicio(df):

    ejec = df.Ejecutada[df.Ejecutada=="Si"].value_counts()
    pend = df.Ejecutada[df.Ejecutada=="No"].value_counts()
    total = np.sum(ejec)+np.sum(pend)
    porc = round(np.sum(pend)/total*100)

    color = obtener_color(porc)

    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Porcentaje de OT pendientes"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{porc}%</span></div>"
    if total == 0:
        return st.write("Periodo no registrado")
    else:
        return st.write(metric_html,unsafe_allow_html=True)
    
def cant_pend_dashboard(df):
    
    pend = df.Ejecutada[df.Ejecutada=="No"].count()

    color = 'color: #808B96; font-size: 50px; text-align: center'
    div_style = "background:#FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Cantidad de OT pendientes"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{pend}</span></div>"
    
    return st.write(metric_html,unsafe_allow_html=True)

def cant_ejec_dashboard(df):    
    ejec = df.Ejecutada[df.Ejecutada=="Si"].count()

    color = 'color: #808B96; font-size: 50px; text-align: center'
    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Cantidad de OT ejecutadas"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{ejec}</span></div>"
    
    return st.write(metric_html,unsafe_allow_html=True)

def graf_ccosto_dashboard_anual(df):
    df = df[df.Año == año_actual]
    conteo_tipo_servicio = df.groupby(["Tipo de Servicio"]).size().reset_index(name="Cantidad")

    fig = px.pie(conteo_tipo_servicio,values="Cantidad",names="Tipo de Servicio",hole=.6)

    fig.update_layout(
        title={
            "text":"Porcentaje de solicitudes por servicio",
            "x":0.432,
            "xanchor": "right"},
        title_font_color= "#D8E2DC",
        font={
            "size":13}, height=350)

    st.plotly_chart(fig,use_container_width=True)

def graf_ccosto_acumulado_mensual_dashboard_anual(df):
    df = df[df.Año == año_actual]
    conteo_ccosto = df.groupby(["Fecha","Nombre CCosto"]).size().reset_index(name="Cantidad")
    cant_ccosto_filtrados = conteo_ccosto.groupby('Nombre CCosto').filter(lambda x: x['Cantidad'].sum() > 40)

    fig = px.bar(cant_ccosto_filtrados, x="Fecha",y="Cantidad",color="Nombre CCosto")

    fig.update_layout(xaxis_title='Fecha',
                      yaxis_title='Cantidad de solicitudes',
                      title={
                          "text":"Cantidad diaria/mensual de solicitudes por centro de costos",
                          "x":0.5,
                          "xanchor": "center"},
                      title_font_color= "#D8E2DC",height=330)

    st.plotly_chart(fig,use_container_width=True) 

def graf_acumulado_servicios_mensual_dashboard(df):
    df = df[df.Año == año_actual]
    df.mes = pd.to_datetime(df.Fecha)
    conteo_servicios = df.groupby([pd.Grouper(key="Fecha",freq="M"),"Tipo de Servicio"]).size().reset_index(name="Cantidad")
    fig = px.line(conteo_servicios, x="Fecha",y="Cantidad",color="Tipo de Servicio",markers=True)

    fig.update_layout(xaxis_title='Fecha',
                      yaxis_title='Cantidad de solicitudes',
                      title={
                          "text":"Cantidad mensual de solicitudes por servicio",
                          "x":0.5,
                          "xanchor": "center"},
                      title_font_color= "#D8E2DC", height=330)
    
    st.plotly_chart(fig,use_container_width=True)

def graf_campus_acum_dashboard(df):
    df = df[df.Año == año_actual]
    conteo_campus = df.groupby(["Ubicación del Trabajo/Servicio"]).size().reset_index(name="Cantidad")
    fig = px.bar(conteo_campus,x="Cantidad",y="Ubicación del Trabajo/Servicio",color="Ubicación del Trabajo/Servicio")

    fig.update_layout(xaxis_title='Cantidad',
                      yaxis_title='Campus',
                      title={
                          "text":"Cantidad de solicitudes por campus",
                          "x":0.5,
                          "xanchor": "center"},
                      title_font_color= "#D8E2DC",height=300)
    
    st.plotly_chart(fig,use_container_width=True)    

def graf_ccosto_dashboard(df):
    conteo_ccosto = df.groupby(["Fecha","Nombre CCosto"]).size().reset_index(name="Cantidad")

    fig = px.bar(conteo_ccosto, x="Fecha", y="Cantidad", color="Nombre CCosto", 
                 barmode='stack',
                 title="Cantidad diaria de OT por centro de costo")

    fig.update_layout(xaxis_title='Fecha',
                      yaxis_title='Cantidad de OT',
                      title={
                          "text": "Cantidad diaria de OT por centro de costo",
                          "x": 0.5,
                          "xanchor": "center"},
                      title_font_color= "#000000",height=330)

    st.plotly_chart(fig, use_container_width=True) 

def graf_ccosto_pie_dashboard(df):
    conteo_ccosto = df.groupby("Nombre CCosto").size().reset_index(name="Cantidad")
    cant_ccosto_filtrado = conteo_ccosto.groupby('Nombre CCosto').filter(lambda x: x['Cantidad'].sum() > 0)

    fig = px.pie(cant_ccosto_filtrado,values="Cantidad",names="Nombre CCosto",hole=.6)

    fig.update_layout(
        title={
            "text":"Distribución de OT por centro de costo",
            "x":0.5,
            "xanchor": "center"},
        title_font_color= "#000000",
        font={
            "size":13}, height=350)

    st.plotly_chart(fig,use_container_width=True)

def graf_dias_dashboard(df):
    conteo_servicio = df.groupby(["Tipo de Servicio","Fecha"]).size().reset_index(name="Cantidad")
    fig = px.line(conteo_servicio, x="Fecha",y="Cantidad",markers=True)

    fig.update_layout(xaxis_title='Fecha',
                      yaxis_title='Cantidad de OT',
                      title={
                          "text":"Cantidad diaria de OT",
                          "x":0.5,
                          "xanchor": "center"},
                      title_font_color= "#000000", height=255)
    
    st.plotly_chart(fig,use_container_width=True)

def graf_pie_campus_dashboard(df):
    conteo_campus = df.groupby(["Ubicación del Trabajo/Servicio"]).size().reset_index(name="Cantidad")

    fig = px.pie(conteo_campus,values="Cantidad",names="Ubicación del Trabajo/Servicio",hole=.5)

    fig.update_layout(
        title={
            "text":"Distribución de OT por campus",
            "x":0.5,
            "xanchor": "center"},
        title_font_color= "#000000",
        font={
            "size":13}, height=320)

    st.plotly_chart(fig,use_container_width=True)    

def mostrar_info_mes_actual(df, año, mes):
    
    df_mes_actual = df[(df['Año'] == año) & (df['Mes'] == mes)]
    total = len(df_mes_actual)
    ejecutadas = len(df_mes_actual[df_mes_actual['Ejecutada'] == 'Si'])
    pendientes = len(df_mes_actual[df_mes_actual['Ejecutada'] == 'No'])
    porc_ejec = round((ejecutadas / total) * 100) if total > 0 else 0
    porc_pend = round((pendientes / total) * 100) if total > 0 else 0

    color = obtener_color(porc_pend)

    div_style = "background: #FFFFFF; padding: 20px; border-radius: 30px; text-align: center;"
    title_style = "font-size: 16px; font-weight: lighter; color: #000000; margin-bottom: 10px;"
    value_style_ejec = f"{color} font-size: 40px;"
    value_style_pend = f"{color} font-size: 40px;"

    metric_html = f"<div style='{div_style}'>"\
                  f"<span style='{title_style}'>Cantidad de OT del mes de {mes}</span></br>"\
                  f"<span style='color: #808B96; font-size: 43px;'>{total}</span></div>"\
                  f"<div style='{div_style}'>"\
                  f"<span style='{title_style}'>Porcentaje de OT ejecutadas</span></br>"\
                  f"<span style='{value_style_ejec}'>{porc_ejec}%</span></div>"\
                  f"<div style='{div_style}'>"\
                  f"<span style='{title_style}'>Porcentaje de OT pendientes</span></br>"\
                  f"<span style='{value_style_pend}'>{porc_pend}%</span></div>"


    return st.write(metric_html, unsafe_allow_html=True)

def grafico_barras_servicios(df, año):
    df = df[(df['Año'] == año)]
    df['Ejecutada'] = df['Ejecutada'].map({'Si': 'Ejecutada', 'No': 'Pendiente'})

    conteo_servicios = df.groupby(['Tipo de Servicio', 'Ejecutada']).size().reset_index(name='Cantidad')

    conteo_pendientes = conteo_servicios[conteo_servicios['Ejecutada'] == 'Pendiente'].sort_values(by='Cantidad', ascending=False)
    servicios_ordenados = conteo_pendientes['Tipo de Servicio'].tolist()

    todos_los_servicios = df['Tipo de Servicio'].unique().tolist()
    servicios_ordenados.extend([servicio for servicio in todos_los_servicios if servicio not in servicios_ordenados])

    conteo_servicios['Tipo de Servicio'] = pd.Categorical(conteo_servicios['Tipo de Servicio'], categories=servicios_ordenados, ordered=True)

    fig = px.bar(conteo_servicios, x='Tipo de Servicio', y='Cantidad', color='Ejecutada',
                 title='Cantidad de OT Ejecutadas y Pendientes por Servicio',
                 labels={'Tipo de Servicio': 'Servicio', 'Cantidad': 'Cantidad de OT'},
                 barmode='group')

    fig.update_layout(
        title={
            'text': 'Cantidad de OT Ejecutadas y Pendientes por Servicio',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title='Servicio',
        yaxis_title='Cantidad de OT',
        legend_title_text='Estado',
        xaxis={'categoryorder': 'array', 'categoryarray': servicios_ordenados}
    )

    return st.plotly_chart(fig, use_container_width=True)

def grafico_barras_mensuales(df, año):
    df = df[(df['Año'] == año)]

    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'])
    
    df['Mes-Año'] = df['Fecha'].dt.to_period('M').dt.strftime('%Y-%m')

    df['Ejecutada'] = df['Ejecutada'].map({'Si': 'Ejecutada', 'No': 'Pendiente'})

    conteo_mensual = df.groupby(['Mes-Año', 'Ejecutada']).size().reset_index(name='Cantidad')

    conteo_mensual = conteo_mensual[conteo_mensual['Cantidad'] > 0]

    fig = px.bar(conteo_mensual, x='Mes-Año', y='Cantidad', color='Ejecutada',
                 labels={'Mes-Año': 'Mes', 'Cantidad': 'Cantidad de OT'},
                 barmode='group')

    fig.update_layout(
            xaxis_title='Mes',
            yaxis_title='Cantidad de OT',
            legend_title_text='Estado',
            xaxis={'categoryorder': 'total ascending'}
    )

    return st.plotly_chart(fig, use_container_width=True)

def filtros_dashboard(df):
    st.sidebar.title("Filtros")

    historico = st.sidebar.checkbox("Resumen anual")
    servicio = st.sidebar.selectbox("Servicio",df["Tipo de Servicio"].unique().tolist())
    año = st.sidebar.selectbox("Año", df["Año"].unique().tolist())
    if año == año_actual:
        if historico:
            return df[(df["Tipo de Servicio"]==servicio)&(df.Año==año_actual)], 1
        mes = st.sidebar.selectbox("Mes", df.Mes[df.Año == año_actual].unique().tolist())
        return df[(df["Tipo de Servicio"]==servicio)&(df.Año==año_actual)&(df.Mes==mes)], 0
    else:
        if historico:
            return df[(df["Tipo de Servicio"]==servicio)&(df.Año==año)], 1
        mes = st.sidebar.selectbox("Mes", df["Mes"].unique().tolist())
        return df[(df["Tipo de Servicio"]==servicio)&(df.Año==año)&(df.Mes==mes)], 0
    
def calcular_promedio_dias(df, año):
    df = df[(df['Año'] == año)]
                       
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha de Término']):
        df['Fecha de Término'] = df['Fecha de Término'].replace(0, pd.NaT)
        df['Fecha de Término'] = pd.to_datetime(df['Fecha de Término'], errors='coerce')

    df['Diferencia en Días'] = (df['Fecha de Término'] - df['Fecha']).dt.days

    df_validos = df[df['Diferencia en Días'] >= 0]

    promedio_dias = df_validos['Diferencia en Días'].mean()

    color = 'color: #808B96; font-size: 50px; text-align: center'
    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Promedio de días OT ejecutadas"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{round(promedio_dias)}</span></div>"

    return st.write(metric_html,unsafe_allow_html=True)

def calcular_promedio_dias_ejec_serv(df):
    # Convertir las columnas de fecha a datetime si no lo son
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha de Término']):
        df['Fecha de Término'] = df['Fecha de Término'].replace(0, pd.NaT)
        df['Fecha de Término'] = pd.to_datetime(df['Fecha de Término'], errors='coerce')

    # Calcular la diferencia en días
    df['Diferencia en Días'] = (df['Fecha de Término'] - df['Fecha']).dt.days

    # Filtrar solo las filas con diferencias en días válidas (no negativas y no NaN)
    df_validos = df[(df['Diferencia en Días'] >= 0) & (df['Diferencia en Días'].notna())]

    # Calcular el promedio de días
    if df_validos.empty:
        promedio_dias = 0
    else:
        promedio_dias = df_validos['Diferencia en Días'].mean()

    color = 'color: #808B96; font-size: 50px; text-align: center'
    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Promedio de días OT ejecutadas"

    metric_html = f"<div style= '{div_style}'>"\
                  f"<span style= '{title_style}'>{titulo}</span></br>"\
                  f"<span style= '{color}'>{round(promedio_dias)}</span></div>"

    return st.write(metric_html, unsafe_allow_html=True)

def calcular_dias_no_ejecutadas(df, año):
    df = df[(df['Año'] == año)]

    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    fecha_actual = pd.to_datetime(datetime.datetime.now().date())

    df['Diferencia en Días No Ejecutadas'] = (fecha_actual - df['Fecha']).dt.days

    df_no_ejecutadas = df[df['Ejecutada'] == 'No']

    promedio_dias_no_ejecutadas = df_no_ejecutadas['Diferencia en Días No Ejecutadas'].mean()

    color = 'color: #808B96; font-size: 50px; text-align: center'
    div_style = "background: #FFFFFF;padding:1px;border-radius:5px;text-align:center;"
    title_style = "font-size:13px;font-weight:lighter;color:#000000;margin-bottom:10px;"
    titulo = "Promedio de días OT No ejecutadas"

    metric_html = f"<div style= '{div_style}'>"\
        f"<span style= '{title_style}'>{titulo}</span></br>"\
        f"<span style= '{color}'>{round(promedio_dias_no_ejecutadas)}</span></div>"
    
    return st.write(metric_html,unsafe_allow_html=True)

def dashboard_anual(df):
    
    df_filtrado(df)

    último_año = df['Fecha'].dt.year.max()
    último_mes = df.Mes.unique().tolist()[0]
    
    size_title = 'font-size: 34px; text-align: center; color: #000000; font-weight: lighter'
    title = f"Indicadores Dirección de Servicios y Logística Año {año_actual}"
    st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)
  
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1:
        porc_pend_dashboard_anual(df, último_año)

    with c2:
        porc_ejec_dashboard_anual(df, último_año)

    with c3:
        cant_pend_dashboard_anual(df, último_año)

    with c4:
        cant_ejec_dashboard_anual(df, último_año)
    with c5:
        calcular_promedio_dias(df, último_año)
    with c6:
        calcular_dias_no_ejecutadas(df,último_año)

    st.write('<br>', unsafe_allow_html=True)

    size_title2 = 'font-size: 28px; text-align: center; color: #000000; font-weight: lighter'
    title2 = f"Estado de OT Ejecutadas y Pendientes Mensual"
    st.write(f'<p style="{size_title2}">{title2}</p>',unsafe_allow_html=True)

    c1,c2 = st.columns([1,3])
    with c1:
        mostrar_info_mes_actual(df, último_año, último_mes)
    with c2:
        grafico_barras_mensuales(df, último_año)

    with st.container():
        grafico_barras_servicios(df, último_año)

def dashboard_personalizado(df):
    df_filtrado(df)

    modo = st.selectbox("¿Qué quieres ver?", ["Indicadores", "Comparar tiempo de ejecución"])
    if modo == "Comparar tiempo de ejecución":
        grafico_tiempo_ejecucion(df)
        return
        
    filtro, titulo = filtros_dashboard(df)

    size_title = 'font-size: 24px; text-align: center; color: #000000; font-weight: lighter'

    if filtro.empty:
        title = "No hay información disponible para el servicio y período seleccionado"
        st.write(f'<p style="{size_title}">{title}</p>', unsafe_allow_html=True)
        return
    
    elif titulo == 1:
            title = f"El servicio de {filtro['Tipo de Servicio'].unique().tolist()[0]} en el año {filtro.Año.unique().tolist()[0]} presenta los siguientes indicadores"
            st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)
    else:
        title = f"El servicio de {filtro['Tipo de Servicio'].unique().tolist()[0]} en el mes de {filtro.Mes.unique().tolist()[0]} del año {filtro.Año.unique().tolist()[0]} presenta los siguientes indicadores"
        st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        porc_no_ejec_dashboard_servicio(filtro)

    with c2:
        porc_ejec_dashboard_servicio(filtro)

    with c3:
        cant_pend_dashboard(filtro)

    with c4:
        cant_ejec_dashboard(filtro)
    
    with c5:
        calcular_promedio_dias_ejec_serv(filtro)

    c1,c2 = st.columns(2)
    with c1:
        graf_ccosto_dashboard(filtro)

    with c2:
        graf_ccosto_pie_dashboard(filtro)

    c1,c2 = st.columns(2)  
    with c1:
        graf_dias_dashboard(filtro)
    with c2:
        graf_pie_campus_dashboard(filtro)

def grafico_tiempo_ejecucion(df):
    # Asegurarnos de que df ya pasó por df_filtrado(df)
    df = df_filtrado(df)

    # 1) Selector de servicio
    servicio = st.selectbox(
        "Selecciona el servicio a comparar",
        options=sorted(df["Tipo de Servicio"].unique())
    )
    df = df[df["Tipo de Servicio"] == servicio]

    # 2) Filtrar sólo las OT ejecutadas
    df_ej = df[df['Ejecutada'] == "Si"].copy()
    df_ej['Fecha'] = pd.to_datetime(df_ej['Fecha'], errors='coerce')
    df_ej['Fecha de Término'] = df['Fecha de Término'].replace(0, pd.NaT)
    df_ej['Fecha de Término'] = pd.to_datetime(df_ej['Fecha de Término'], errors='coerce')

    # 3) Calcular días de ejecución
    df_ej['DiasEjec'] = (df_ej['Fecha de Término'] - df_ej['Fecha']).dt.days

    # filtra los negativos
    df_ej = df_ej[df_ej['DiasEjec'] >= 0]

    # 4) Extraer año y mes numérico
    df_ej['Año']     = df_ej['Fecha'].dt.year
    df_ej['MesNum']  = df_ej['Fecha'].dt.month

    # 5) Mapear nombres de mes en español
    dicc_meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
        5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
        9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre"
    }
    df_ej['Mes'] = df_ej['MesNum'].map(dicc_meses)

    # 6) Agrupar y calcular promedio de días
    grp = (
        df_ej
        .groupby(['Año', 'MesNum', 'Mes'])['DiasEjec']
        .mean()
        .reset_index(name='PromedioDias')
    )

    # 7) Asegurar orden cronológico de meses
    meses_ordenados = list(dicc_meses.values())
    grp['Mes'] = pd.Categorical(grp['Mes'], categories=meses_ordenados, ordered=True)

    # 8) Dibujar línea
    fig = px.line(
        grp.sort_values(['Año','MesNum']),
        x='Mes', y='PromedioDias', color='Año', markers=True,
        title=f"Promedio de días de ejecución para «{servicio}»",
        labels={'PromedioDias': 'Días promedio', 'Mes': 'Mes'}
    )
    fig.update_layout(xaxis_title='Mes', yaxis_title='Días promedio de ejecución')

    st.plotly_chart(fig, use_container_width=True)

def df_filtrado (df):
    cols_a_quitar = [
        "Adj.", "Resumen", "Detalle", "Anexo", "Funcionario de Contacto_cod",
        "Funcionario de Contacto_dsc", "Fecha de Recepción", "Funcionario Encargado_cod",
        "Funcionario Encargado_dsc", "Fecha de Asignación", "Funcionario de Contacto_cod",
        "Solicitud de Compra", "Observación", "Nº de Horas Hombre", "Cantidad de Personas Involucradas",
        "Material Utilizado", "Rut Responsable_cod", "Rut Responsable_dsc", "Funcionario Ejecutor_cod",
        "Funcionario Ejecutor_dsc", "Ubicación Específica", "Ubicación", "Fecha y Hora Sistema"
    ]

    # Le decimos a pandas que ignore (sin lanzar KeyError) si alguna no existe
    df.drop(columns=cols_a_quitar, inplace=True, errors='ignore')

    # Rellena NaN con ceros (sin argumentos extras)
    df.fillna(0, inplace=True)

    if not pd.api.types.is_datetime64_any_dtype(df['Fecha']):
        df['Fecha'] = df['Fecha'].replace(0, pd.NaT)
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')

    df["Ejecutada"] = df["Fecha de Término"].apply(lambda x: "No" if x == 0 else "Si")
    dicc_meses = {
        1:"Enero",
        2:"Febrero",
        3:"Marzo",
        4:"Abril",
        5:"Mayo",
        6:"Junio",
        7:"Julio",
        8:"Agosto",
        9:"Septiembre",
        10:"Octubre",
        11:"Noviembre",
        12:"Diciembre"}

    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    df.Mes = df.Mes.map(dicc_meses,na_action="ignore")
    
    cambiar_nombres = {"Centro de Costo_cod":"Código CCosto","Centro de Costo_dsc":"Nombre CCosto"}
    
    df.rename(columns=cambiar_nombres, inplace=True)
    
    return df

def matriz_corr(df):
    pass
    # matriz_correlacion = df[["Nº de Solicitud","Centro de Costo_cod","Solicitud de Compra","Nº de Horas Hombre","Cantidad de Personas Involucradas"]].corr()
    # fig, ax = plt.subplots()
    # sns.heatmap(matriz_correlacion, annot=True, ax=ax)
    # st.pyplot(fig)

def distribucion_col_num(df):
    df = df.drop("Adj.", axis=1)
    df = df.drop("Rut Responsable_dsc",axis=1)
    size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
    title = "Distribución de columnas numéricas"
    st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)
    for col in df.select_dtypes(include=np.number):
        fig = px.histogram(df, x=col)
        fig.update_layout(title={
                                "text":f"Distribución de {col}",
                                "x":0.5,
                                "xanchor": "center"},
                      title_font_color= "#D8E2DC")#, height=255)        
        st.plotly_chart(fig,use_container_width=True)

def distribucion_col_categoricas(df):
    df = df.drop(["Resumen","Detalle","Ubicación","Anexo","Funcionario de Contacto_cod","Funcionario de Contacto_cod",
                  "Funcionario de Contacto_dsc","Fecha de Recepción","Funcionario Encargado_cod","Funcionario Encargado_dsc",
                  "Funcionario Ejecutor_cod","Fecha de Término","Observación","Material Utilizado","Rut Responsable_cod","Rut Responsable_dsc"],axis=1)
   
    size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
    title = "Frecuencia de columnas categóricas"
    st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)
    for col in df.select_dtypes(include=object):
        fig = px.histogram(df, x=col)
        fig.update_layout(title={
                                "text":f"Frecuencia de {col}",
                                "x":0.5,
                                "xanchor": "center"},
                      title_font_color= "#D8E2DC")#, height=255)  
        st.plotly_chart(fig,use_container_width=True)

def pie_valores_nulos(df):
    fig = px.pie(df.isnull().sum(),values=df.isnull().sum().values,names=df.isnull().sum().index.tolist(),hole=0.3)     
    st.plotly_chart(fig,use_container_width=True)

def bar_cantidad_nulos(df):
    fig = px.bar(df.isnull().sum(),x=df.isnull().sum().values,y=df.isnull().sum().index.tolist())      
    st.plotly_chart(fig,use_container_width=True) 

def grafico_barras(df):
    
    df_filtrado(df)
    
    st.title('Generador de Gráficos de Barras')
    
    tipo_grafico = st.selectbox('Selecciona el tipo de gráfico', options=['Personalizado', 'Histórico por servicio', 'Histórico por año'])
    
    valores_columnas = df.columns.tolist()
    
    if tipo_grafico == 'Personalizado':
        x_col = st.selectbox('Selecciona la columna para el eje X', options=valores_columnas)
        y_col = st.selectbox('Selecciona la columna para el eje Y', options=valores_columnas)
        
        if st.button('Generar gráfico'):
            counts = df[x_col].value_counts().nlargest(10).reset_index()
            counts.columns = [x_col, y_col]
            
            st.markdown(f"## Gráfico de barras de {x_col}")
            fig = px.bar(counts, x=x_col, y=y_col, labels={'index': x_col}, title=f'Gráfico de barras de {x_col}')
            st.plotly_chart(fig)
    
    elif tipo_grafico == 'Histórico por servicio':
        conteo_servicios = df.groupby(['Año', 'Tipo de Servicio', 'Ejecutada']).size().reset_index(name='Cantidad')
        df['Ejecutada'] = df['Ejecutada'].map({'Si': 'Ejecutada', 'No': 'Pendiente'})
        
        if st.button('Generar gráfico'):
            fig = px.bar(conteo_servicios, x='Tipo de Servicio', y='Cantidad', color='Ejecutada', facet_col='Año',
                         title='Cantidad de OT histórico por servicio',
                         labels={'Tipo de Servicio': 'Servicio', 'Cantidad': 'Cantidad de OT'},
                         barmode='group')
            
            fig.update_layout(
                title={
                    'text': 'Cantidad de OT histórico por servicio',
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Servicio',
                yaxis_title='Cantidad de OT',
                legend_title_text='Estado',
                xaxis={'categoryorder': 'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif tipo_grafico == 'Histórico por año':
        conteo_anual = df.groupby(['Año', 'Ejecutada']).size().reset_index(name='Cantidad')
        df['Ejecutada'] = df['Ejecutada'].map({'Si': 'Ejecutada', 'No': 'Pendiente'})
        
        if st.button('Generar gráfico'):
            fig = px.bar(conteo_anual, x='Año', y='Cantidad', color='Ejecutada',
                         title='Cantidad de OT histórico por año',
                         labels={'Año': 'Año', 'Cantidad': 'Cantidad de OT'},
                         barmode='group')
            
            fig.update_layout(
                title={
                    'text': 'Cantidad de OT histórico por año',
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                xaxis_title='Año',
                yaxis_title='Cantidad de OT',
                legend_title_text='Estado',
                xaxis={'categoryorder': 'total descending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)

def grafico_boxplot(df):
    df.drop(columns=["Adj.","Fecha de Recepción","Funcionario Encargado_cod","Funcionario Encargado_dsc","Fecha de Asignación","Funcionario Ejecutor_cod","Funcionario Ejecutor_dsc","Solicitud de Compra","Observación","Material Utilizado","Rut Responsable_dsc"],inplace=True)
    

    st.title('Generador de Gráficos de Líneas')
    valores_columnas = ['Centro de Costo_dsc', 'Nº de Solicitud']
    groupby_columna=['Ubicación del Trabajo/Servicio','Tipo de Trabajo','Nº de Horas Hombre','Cantidad de Personas Involucradas']
    
    # Solicitar al usuario el valor de la columna y la columna para agrupar
    valor_columna = st.selectbox("Selecciona la columna de valor", valores_columnas)
    groupby_x = st.selectbox("Selecciona la columna para agrupar", groupby_columna)
    # Generar el gráfico de cajas utilizando plotly.express
    fig = px.box(df, y=valor_columna, x=groupby_x)

    # Mostrar el gráfico utilizando Streamlit
    st.plotly_chart(fig)

def grafico_lineas(df):
    
    df = df_filtrado(df)
    
    st.title('Generador de Gráficos de Líneas')
    
    # Selección del tipo de servicio
    servicios_disponibles = df['Tipo de Servicio'].unique()
    servicio_seleccionado = st.selectbox('Selecciona el tipo de servicio', options=servicios_disponibles)
    
    # Filtrar el DataFrame por el tipo de servicio seleccionado
    df_servicio_seleccionado = df[df['Tipo de Servicio'] == servicio_seleccionado]
    
    # Contar las órdenes por mes y año
    conteo_mensual = df_servicio_seleccionado.groupby(['Año', 'Mes']).size().reset_index(name='Cantidad')
    
    # Ordenar los meses según su orden en el año
    meses_ordenados = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    conteo_mensual['Mes'] = pd.Categorical(conteo_mensual['Mes'], categories=meses_ordenados, ordered=True)
    
    # Crear una columna para ordenar por año y mes
    conteo_mensual['Año-Mes'] = conteo_mensual.apply(lambda row: f"{row['Año']}-{meses_ordenados.index(row['Mes']) + 1:02d}", axis=1)
    conteo_mensual = conteo_mensual.sort_values(by=['Año-Mes'])
    
    # Crear el gráfico de líneas
    fig = px.line(conteo_mensual, x='Mes', y='Cantidad', color='Año',
                  title=f'Cantidad de OT histórico por Mes para el servicio de {servicio_seleccionado}',
                  labels={'Mes': 'Mes', 'Cantidad': 'Cantidad de OT'}, markers=True)
    
    fig.update_layout(
        xaxis_title='Mes',
        yaxis_title='Cantidad de OT',
        title={
            'text': f'Cantidad de OT histórico por Mes para el servicio de {servicio_seleccionado}',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)

def grafico_histograma(df):
    st.title('Generador de Gráficos de Histograma')

    columnas = ['Nº de Solicitud', 'Fecha', 'Tipo de Servicio', 'Nº de Horas Hombre', 
                'Cantidad de Personas Involucradas', 'Material Utilizado', 'Ubicación del Trabajo/Servicio']

    # Seleccionar una columna para el gráfico de histograma
    columna_graficar = st.selectbox('Elige una columna para generar el histograma', options=columnas)

    # Filtrar valores nulos en la columna seleccionada
    df_filtrado = df[df[columna_graficar].notnull()]

    # Generar el gráfico de histograma
    fig = px.histogram(df_filtrado, x=columna_graficar, title=f'Histograma de {columna_graficar}')

    # Mostrar el gráfico
    st.plotly_chart(fig)

def grafico_pastel(df):
    st.title('Generador de Gráfico de Pastel')

    # Definir las columnas adecuadas para el gráfico de pastel
    columnas = ['Tipo de Servicio', 'Centro de Costo_dsc', 'Funcionario Encargado_dsc', 'Ubicación del Trabajo/Servicio']

    # Seleccionar una columna para el gráfico de pastel
    columna_graficar = st.selectbox('Elige una columna para el gráfico de pastel', options=columnas)

    if columna_graficar:
        valores = df[columna_graficar].value_counts().head(10)
        fig = px.pie(valores, values=valores, names=valores.index, title=f'Distribución de los 10 más importantes en {columna_graficar}')
        st.plotly_chart(fig)
    else:
        st.warning('No hay columnas disponibles para generar el gráfico de pastel.')

def crear_grafico (tipo,df):

    df.fillna(0,inplace=True)

    if (len(tipo)>=0):
        
        if tipo == "Barras":
            grafico_barras(df)

        if tipo == "Líneas":
           grafico_lineas(df)

        if tipo == "BoxPlot":
           grafico_boxplot(df)

        if tipo == "Histograma":
           grafico_histograma(df)
        
        if tipo =="Pastel":
            grafico_pastel(df)

    else:
        st.write("Algo Salio mal vuelve a intentarlo")

def analisis_exp(df):
    size_title = 'font-size: 40px; text-align: center; color: #D8E2DC; font-weight: lighter'
    title = "Análisis explotario de la base de datos ingresada"
    st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True) 
    size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
    title = "Cabecera de la base de datos"
    st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True) 
    st.write(df.head(5))
    size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
    title = "Descripción estadística de la base de datos"
    st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True) 
    st.write(df.describe())
    c1,c2 = st.columns(2)
    with c1:
        size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
        title = "Datos faltantes por columna"
        st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True) 
        df_nulos(df)
    with c2:
        size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
        title = "Correlación de datos por columna"
        st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True) 
        matriz_corr(df)
    c1,c2 = st.columns(2)
    with c1:
        size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
        title = "Porcentaje de valores nulos por columna"
        st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True) 
        pie_valores_nulos(df)
    with c2:
        size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
        title = "Cantidad de valores nulos por columna"
        st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)        
        bar_cantidad_nulos(df)
    c1,c2 = st.columns(2)
    with c1:
        distribucion_col_num(df)
        size_title = 'font-size: 30px; text-align: center; color: #D8E2DC; font-weight: lighter'
        title = "Tipos de datos por columna"
        st.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)       
        st.dataframe(df.dtypes)
    with c2:
        distribucion_col_categoricas(df)

def principal():
    size_title = 'font-size: 24px; text-align: center; color: #000000; font-weight: lighter'
    title = "Dashboard para análisis de órdenes de trabajo de la DSL"
    st.sidebar.write(f'<p style="{size_title}">{title}</p>',unsafe_allow_html=True)
    #st.sidebar.write("Seleccione una base de datos")
    #archivo_excel = st.sidebar.file_uploader("Elija archivo Excel",type=["xlsx"])
    bd_default = "ot_2019_2025_mayo.xlsx"
    df = cargar_datos(bd_default)
    archivo_excel = bd_default
    if archivo_excel is None:
        opciones = st.sidebar.radio(" ", options=["Panel Principal","Gráficos Personalizados","Generador de Gráficos"])
        
        if opciones == "Gráficos Personalizados":
            dashboard_personalizado(df)

        elif opciones == "Panel Principal":
            dashboard_anual(df)

        elif opciones == "Análisis Exploratorio":
            analisis_exp(df)

        elif opciones == "Generador de Gráficos":

            tipo_grafico = st.sidebar.selectbox("Seleccione el tipo de gráfico", ["Barras","Líneas","BoxPlot","Histograma","Pastel"])
            crear_grafico(tipo_grafico,df)

    if archivo_excel is not None:
        opciones = st.sidebar.radio(" ", options=["Panel Principal","Gráficos Personalizados","Generador de Gráficos"])
        df_dsl = cargar_datos(archivo_excel)
        
        if opciones == "Gráficos Personalizados":
            dashboard_personalizado(df_dsl)

        elif opciones == "Panel Principal":
            dashboard_anual(df_dsl)

        elif opciones == "Análisis Exploratorio":
            analisis_exp(df_dsl)

        elif opciones == "Generador de Gráficos":
            tipo_grafico = st.sidebar.selectbox("Seleccione el tipo de gráfico", ["Barras","Líneas","BoxPlot","Histograma","Pastel"])
            crear_grafico(tipo_grafico,df_dsl)
            
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    principal()

else:
    mostrar_login()
