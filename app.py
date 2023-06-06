import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from pycaret.regression import *
from sklearn.pipeline import Pipeline
import numpy as np
import gdown

#configuração da página
st.set_page_config(page_title="Desmatamento", page_icon="🌳", layout="centered", initial_sidebar_state="collapsed")

#definição do tema
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #4F4F4F;
}
</style>
    """, unsafe_allow_html=True)

#título
st.title("Como serão os números de desmatamento agora na gestão do Lula em 2023? E como seria se o Bolsonaro tivesse sido eleito?")

#exibir imagem tema do lula
st.image("desmatamento.png")

#texto
st.markdown("Há muita expectativa com a gestão do Lula no que se diz respeito ao desmatamento, já que este foi um dos principais alvos da campanha de Lula contra Bolsonaro nos debates eleitorais. O objetivo do projeto é entender os números gerais do desmatamento da Amazônia Legal ao longo dos últimos anos, entender qual foi impacto do governo Bolsonaro nestes números e prever tanto como será o desmatamento para 2023 na gestão de Lula, como prever caso o Bolsonaro tivesse sido eleito.")

#1º Bloco************************************************************************************************************************
st.subheader("Análises Preliminares")

st.markdown("O primeiro passo do estudo é entender os números gerais de incremento de desmatamento, analisando por estado, ano e governo. Os dados consideram incrementos a partir de 2008.")

#impor do arquivo
df = pd.read_csv("desmatamento.csv")
#df['date'] = pd.to_datetime(df['date'])
#st.write(df.columns)

#Marcação do governo de acordo com o ano
def marcar_governo(year):
    if year < 2011:
        return 'Lula'
    elif 2011 <= year <= 2014:
        return 'Dilma1'
    elif 2015 <= year < 2017:
        return 'Dilma2'
    elif 2017 <= year <= 2018:
        return 'Temer'
    elif 2018 <= year <= 2022:
        return 'Bolsonaro'
    else:
        return 'Desconhecido' 
    
df['governo'] = df['year'].apply(marcar_governo)


# Ordenar os dados por estado, município, geocode_ibge e ano
df = df.sort_values(['state', 'municipality', 'geocode_ibge', 'year'])

# Calcular a diferença ano a ano no desmatamento por cada combinação única de estado, município e geocode_ibge
df['delta_areakm'] = df.groupby(['state', 'municipality', 'geocode_ibge'])['areakm'].diff()

# Preencher NA/NaN valores com 0s (isso ocorrerá para o primeiro ano de cada combinação única, pois não há ano anterior para comparar)
df['delta_areakm'] = df['delta_areakm'].fillna(0)

df_grafico1 = df.copy()

#divindo a página em dois blocos
col1, col2, col3 = st.columns(3)

#Filtro de Estado------------------------------------
#criando um widget de seleção para selecionar o estado
estados = df_grafico1['state'].unique()
#estados = [estados.title() for estado in estados] 
with col1:
    estado_selecionado = st.selectbox('Selecione um estado:', options=['Todos'] + list(estados))

    #filtrando os dados pelo estado selecionado
    if estado_selecionado != 'Todos':
        df_grafico1 = df_grafico1[df_grafico1['state'] == estado_selecionado]

with col2:

    #Filtro de Ano------------------------------------
    #criando um widget de seleção para selecionar o estado
    anos = df_grafico1['year'].unique()
    ano_selecionado = st.selectbox('Selecione um ano:', options=['Todos'] + list(anos))

    #filtrando os dados pelo ano selecionado
    if ano_selecionado != 'Todos':
        df_grafico1 = df_grafico1[df_grafico1['year'] == ano_selecionado]

with col3:

    #Filtro de Ano------------------------------------
    #criando um widget de seleção para selecionar o estado
    governos = df_grafico1['governo'].unique()
    governo_selecionado = st.selectbox('Selecione um governo:', options=['Todos'] + list(governos))

    #filtrando os dados pelo ano selecionado
    if governo_selecionado != 'Todos':
        df_grafico1 = df_grafico1[df_grafico1['governo'] == governo_selecionado]

col1, col2 = st.columns(2)

#Primeiro gráfico -----------------------------------------------------------------------
grupo_uf = df_grafico1.groupby('state')['delta_areakm'].sum().reset_index()
grupo_uf = grupo_uf.sort_values(by=['delta_areakm'])

# Usando a primeira coluna para o gráfico
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.barh(grupo_uf['state'], grupo_uf['delta_areakm'])
    plt.title('Total Incremento de Desmatamento por Estado', fontsize=15)
    plt.xlabel('Área em Kms')
    #plt.ylabel('Total de Queimadas')
    plt.ticklabel_format(style='plain', axis='x')
    plt.tight_layout()
    st.pyplot(fig1)
    
#Segundo gráfico -----------------------------------------------------------------------
    
#agrupando os dados pelo mês e calculando a soma do campo 'focuses'
# grouped_df = df_grafico1.groupby('governo')['areakm'].sum().reset_index()

# #criando uma nova figura para o gráfico
# fig2 = plt.figure(figsize=(6,5))

# #criando um gráfico de barras
# sns.barplot(x=grouped_df['governo'], y=grouped_df['areakm'], color='blue')

# #usando a primeira coluna para o gráfico
# with col2:
#     fig2, ax2 = plt.subplots(figsize=(6, 5))
#     ax2.bar(grouped_df['governo'], grouped_df['areakm'])
#     plt.title('Total Desmatamentos por Ano', fontsize=15)
#     plt.xlabel('Mês')
#     plt.tight_layout()
#     st.pyplot(fig2)
    

#Terceiro gráfico -----------------------------------------------------------------------

#agrupando os dados pela data e calcule a soma do campo 'focuses'
with col2:
    df_grafico1['year_tratado'] = df_grafico1['year'].astype(str).str.slice(-2)
    grouped_df = df_grafico1.groupby('year_tratado')['delta_areakm'].sum().reset_index()
    grouped_df = grouped_df[grouped_df['year_tratado'] != '07']

    #definindo o estilo Seaborn
    sns.set_theme()

    #cria um gráfico de linha
    fig3, ax3 = plt.subplots(figsize=(6,5))
    ax3.plot_date(grouped_df['year_tratado'], grouped_df['delta_areakm'], linestyle='solid', color='blue')
    ax3.set_title('Histórico Incremento de Desmatamento por Ano', fontsize=15)
    plt.tight_layout()
    st.pyplot(fig3)
    

#df_grafico1['year_tratado'] = df_grafico1['year'].astype(str).str.slice(-2)
grouped_df = df_grafico1.groupby('governo')['delta_areakm'].sum().reset_index()
#grouped_df = grouped_df[grouped_df['year_tratado'] != '07']

#definindo o estilo Seaborn
sns.set_theme()

# Criando um dicionário com cores para cada governo
cores = {'Lula': 'red', 'Dilma1': 'pink','Dilma2': 'orange', 'Temer': 'gray', 'Bolsonaro': 'green'}


# Ordenando o DataFrame pela ordem desejada
order = ['Lula', 'Dilma1', 'Dilma2', 'Temer', 'Bolsonaro']
grouped_df['governo'] = pd.Categorical(grouped_df['governo'], categories=order, ordered=True)
grouped_df = grouped_df.sort_values('governo')

# Criando um gráfico de barras
fig3, ax3 = plt.subplots(figsize=(10,6))
barras = ax3.bar(grouped_df['governo'], grouped_df['delta_areakm'], color=[cores[i] for i in grouped_df['governo']])
ax3.set_title('Histórico Incremento de Desmatamento por Governo', fontsize=15)

# Adicionando rótulos nas barras
for bar in barras:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

plt.tight_layout()
st.pyplot(fig3)


#texto
st.markdown("""Alguns pontos chamam muito atenção nestes gráficos. O primeiro é a representatividade do Pará nos incrementos de áreas desmatadas. De fato ele é o segundo estado que mais abrange a área da Amazônia Legal, mas ele possui mais do que o triplo de área desmatada do Amazonas, que é o estado que mais possui área da Amazônia Legal. Isso provavelmente se deve a facilidade de exploração no estado do Pará em comparação com o Amazonas, por conta do maior desenvolvimento da região.
            
Outro ponto de destaque é a evolução de área desmatada ao longo do tempo. Percebemos um pico de incremento em 2008, no governo Lula, e depois disso tivemos seguidos anos de queda até 2012, quando os números começaram a crescer de forma mais tímida no segundo ano de governo da Dilma. Este ritmo se repetiu durante alguns anos.
            
Com o Temer os números se estabilizaram. E a partir de 2019, com o governo Bolsonaro, houve um nítido aumento agressivo dos incrementos de desmatamento. Tanto que se somarmos os incrementos do governo Dilma com os de Temer, ficamos apenas um pouco acima dos números que tivemos no governo de Bolsonaro. Ou seja, o que perdemos de Amazônia Legal em um período de 8 anos com Dilma e Temer foi praticamente o mesmo que perdemos em 4 anos de mandato de Bolsonaro.""")

# #2º Bloco************************************************************************************************************************
st.subheader("Análises dos números dos Estados por Governo Presidencial")



# Agrupar os dados por estado e governo
grouped_df = df.groupby(['state', 'governo'])['delta_areakm'].sum().reset_index()

# Reorganizando os dados para o formato que o matplotlib precisa para barras empilhadas
grouped_df_pivot = grouped_df.pivot(index='state', columns='governo', values='delta_areakm').reset_index()
grouped_df_pivot = grouped_df_pivot.set_index('state')

# Definindo o estilo Seaborn
sns.set_theme()

# Definindo as cores para cada governo
cores = {'Lula': 'red', 'Dilma1': 'pink','Dilma2': 'orange', 'Temer': 'gray', 'Bolsonaro': 'green'}

# Criando o gráfico de barras empilhadas
fig3, ax3 = plt.subplots(figsize=(10,6))
grouped_df_pivot.loc[:, order].plot(kind='bar', stacked=True, color=[cores[i] for i in order], ax=ax3)
ax3.set_title('Representatividade de cada Governo no Desmatamento por Estado', fontsize=15)
plt.tight_layout()

# Mostrar o gráfico
st.pyplot(fig3)


#texto
st.markdown("""Acima podemos observar a representativade que o governo de Bolsonaro teve nos incrementos de cada Estado. Em praticamente todos os estados ele teve a maior parcela de incremento em relação a outros governos, com exceção do Maranhão.""")

# 3º Bloco************************************************************************************************************************


#Projeão Lula

# #texto
st.subheader("Modelo preditivo")

st.markdown("Por fim, vamos projetar como será o ano de 2023 com o Lula e como seria caso Bolsonaro fosse eleito.")


if st.button("Projetar"):
    df_projecao_aux = df.copy()
    df_projecao = df_projecao_aux[['municipality','geocode_ibge','state','governo']].drop_duplicates()
    df_projecao = df_projecao.drop(columns=['governo'])
    df_projecao['year'] = 2023
    df_projecao['governo_Lula'] = 1
    df_projecao['governo_Dilma'] = 0
    df_projecao['governo_Temer'] = 0
    df_projecao['governo_Bolsonaro'] = 0
    df_projecao_dummies = pd.get_dummies(df_projecao, columns=['state', 'municipality'])
    
    
    

    #url = 'https://drive.google.com/uc?export=download&id=1pqnJpGAkOlPucPYSMvT2rcdQPM1PId8-'
    #output = 'pycaret_mdl_rf.pkl'
    #gdown.download(url, output, quiet=False)
    #mdl_et = load_model('pycaret_mdl_rf')
    mdl_xgboost = load_model('./pycaret_mdl_xg')
    xgboost_model = mdl_xgboost.named_steps['trained_model']

    ypred = predict_model( xgboost_model, data = df_projecao_dummies)
    df_projecao_dummies['delta_areakm'] = ypred['prediction_label']


    # Agrupa os dados por ano e soma os valores de 'areakm'
    df_projecao_dummies = df_projecao_dummies.drop_duplicates()
    df_projecao_agrupado = df_projecao_dummies.groupby('year')['delta_areakm'].sum().reset_index()
    df_agrupado = df.groupby('year')['delta_areakm'].sum().reset_index()

    # Concatena os dois DataFrames
    comparativo_ano = pd.concat([df_agrupado, df_projecao_agrupado], ignore_index=True)
    comparativo_ano = comparativo_ano[comparativo_ano['year'] != 2007]
    # Cria o gráfico de barras
    fig8, ax8 = plt.subplots(figsize=(10,6))

    # Define a largura das barras
    bar_width = 0.2

    # Define uma cor padrão e uma cor destacada
    default_color = 'blue'
    highlight_color = 'red'

    # Cria o gráfico para cada ano
    for i, ano in enumerate(comparativo_ano['year'].unique()):
        df_ano = comparativo_ano[comparativo_ano['year'] == ano]
        # Adiciona um valor constante ao argumento `x` do método `bar` para ajustar a posição das barras
        bars = ax8.bar(i + bar_width, df_ano['delta_areakm'], width=bar_width, 
                       color=highlight_color if ano == 2023 else default_color)

        # Adiciona rótulos nas barras
        for bar in bars:
            yval = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    # Legendas e títulos
    ax8.set_xlabel('Ano')
    ax8.set_ylabel('Área desmatada (km²)')
    ax8.set_title('Desmatamento por ano')

    # Ajusta os ticks do eixo X para corresponderem ao meio das barras e define os rótulos dos ticks como os anos
    ax8.set_xticks(np.arange(len(comparativo_ano['year'].unique())) + bar_width)
    ax8.set_xticklabels(comparativo_ano['year'].unique())

    plt.tight_layout()
    st.pyplot(fig8)

    st.markdown("A projeção indica que este primeiro ano de governo Lula irá trazer os números de incremento de desmatamento para patamares um pouco menores do que o primeiro ano do governo Bolsonaro. Agora vamos avaliar como seria a projeção caso o Bolsonaro tivesse sido eleito para este ano.")



    #Projeão Bolsonaro

    # #texto
    #st.subheader("Modelo preditivo")

    df_projecao_aux = df.copy()
    df_projecao = df_projecao_aux[['municipality','geocode_ibge','state','governo']].drop_duplicates()
    df_projecao = df_projecao.drop(columns=['governo'])

    df_projecao['year'] = 2023
    df_projecao['governo_Lula'] = 0
    df_projecao['governo_Dilma'] = 0
    df_projecao['governo_Temer'] = 0
    df_projecao['governo_Bolsonaro'] = 1

    df_projecao_dummies = pd.get_dummies(df_projecao, columns=['state', 'municipality'])

    # mdl_et = load_model('./pycaret_mdl_rf')
    # et_model = mdl_et.named_steps['trained_model']

    ypred = predict_model(xgboost_model, data = df_projecao_dummies)
    df_projecao_dummies['delta_areakm'] = ypred['prediction_label']


    # Agrupa os dados por ano e soma os valores de 'areakm'
    df_projecao_dummies = df_projecao_dummies.drop_duplicates()
    df_projecao_agrupado = df_projecao_dummies.groupby('year')['delta_areakm'].sum().reset_index()
    df_agrupado = df.groupby('year')['delta_areakm'].sum().reset_index()

    # Concatena os dois DataFrames
    comparativo_ano = pd.concat([df_agrupado, df_projecao_agrupado], ignore_index=True)
    comparativo_ano = comparativo_ano[comparativo_ano['year'] != 2007]
    # Cria o gráfico de barras
    fig8, ax8 = plt.subplots(figsize=(10,6))

    # Define a largura das barras
    bar_width = 0.2

    # Define uma cor padrão e uma cor destacada
    default_color = 'blue'
    highlight_color = 'red'

    # Cria o gráfico para cada ano
    for i, ano in enumerate(comparativo_ano['year'].unique()):
        df_ano = comparativo_ano[comparativo_ano['year'] == ano]
        # Adiciona um valor constante ao argumento `x` do método `bar` para ajustar a posição das barras
        bars = ax8.bar(i + bar_width, df_ano['delta_areakm'], width=bar_width, 
                       color=highlight_color if ano == 2023 else default_color)

        # Adiciona rótulos nas barras
        for bar in bars:
            yval = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    # Legendas e títulos
    ax8.set_xlabel('Ano')
    ax8.set_ylabel('Área desmatada (km²)')
    ax8.set_title('Desmatamento por ano')

    # Ajusta os ticks do eixo X para corresponderem ao meio das barras e define os rótulos dos ticks como os anos
    ax8.set_xticks(np.arange(len(comparativo_ano['year'].unique())) + bar_width)
    ax8.set_xticklabels(comparativo_ano['year'].unique())

    plt.tight_layout()
    st.pyplot(fig8)

    st.markdown("A principío esta projeção gera estranhamento já que o esperado seria um contínuo aumento do que vinhamos tendo nos governos anteriores de Bolsonaro, considerando a pouca importância que a pauta de desmatamento teve ao longo do governo e os aumentos sucessivos de incrementos a cada ano. Contudo, o modelo não considera apenas a variável de quem é o presidente mas também o comportamento histórico dos incrementos. E se notarmos ao longo dos anos não tivemos nenhum período com 4 anos de aumento consecutivo, portanto, o valor apresentado pode indicar apenas um leve recuo para a projeção se moldar de um modo que faça mais sentido a curva histórica. ")








    

        
