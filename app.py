import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from pycaret.regression import *
from sklearn.pipeline import Pipeline


#configuração da página
st.set_page_config(page_title="As queimadas associadas a desmatamento estão aumentando na gestão do Lula?", page_icon=":smiley:", layout="centered", initial_sidebar_state="collapsed")

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
st.title("As queimadas associadas ao desmatamento estão aumentando na gestão do Lula?")

#exibir imagem tema do lula
st.image("lula_queimada.png")

#texto
st.markdown("Durante as eleições, as questões ambientais, em especial o desmatamento, estiveram entre os temas mais discutidos. Essa questão foi alvo de críticas intensas durante a gestão de Bolsonaro, e figurou como uma das principais promessas de campanha de Lula. Alguns meses se passaram desde então, como está a situação agora? O propósito deste estudo é avaliar um ponto específico do desmatamento, os números gerais de queimadas associadas ao recente desflorestamento. Embora as queimadas possam ter algumas causas que escapam do controle das políticas públicas, sabemos que parte está ligada às ações intencionais do homem, principalmente com a finalidade de usar as terras para atividades agrícolas ou pastoris. Dessa forma, pretendemos analisar e contrastar os números de queimadas durante a gestão de ambos os presidentes. Além disso, pretendemos utilizar técnicas de Machine Learning para prever os números de queimadas no mês seguinte, contribuindo para estratégias de prevenção e controle desses incidentes.")

#1º Bloco************************************************************************************************************************
st.subheader("Análises Preliminares - Queimadas")

st.markdown("O primeiro passo do estudo é entender os números gerais das queimadas por estado, ao longo dos últimos meses e a representatividade que cada mês possui nas queimadas para avaliarmos a sazonalidade. Abaixo há um filtro de estado que permite verificar dados de localidades específcas.")

df = pd.read_csv("dados_queimadas.csv",delimiter=";")
#convertendo 'date' para datetime
df['date'] = pd.to_datetime(df['date'])

#criando um widget de seleção para selecionar o estado
states = df['uf'].unique()
states = [state.title() for state in states] 
selected_state = st.selectbox('Selecione um estado:', options=['Todos'] + list(states))

#filtrando os dados pelo estado selecionado
if selected_state != 'Todos':
    df = df[df['uf'].str.title() == selected_state]
    
#divindo a página em dois blocos
col1, col2 = st.columns(2)

#extraindo o mês
df['mes'] = df['date'].dt.month


#Primeiro gráfico -----------------------------------------------------------------------
grupo_uf = df.groupby('uf')['focuses'].sum().reset_index()
grupo_uf = grupo_uf.sort_values(by=['focuses'])

# Usando a primeira coluna para o gráfico
with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.barh(grupo_uf['uf'], grupo_uf['focuses'])
    plt.title('Total Queimadas por Estado', fontsize=15)
    plt.xlabel('Mês')
    #plt.ylabel('Total de Queimadas')
    plt.tight_layout()
    st.pyplot(fig1)
    
#Segundo gráfico -----------------------------------------------------------------------
    
#agrupando os dados pelo mês e calculando a soma do campo 'focuses'
grouped_df = df.groupby('mes')['focuses'].sum().reset_index()

#criando uma nova figura para o gráfico
fig2 = plt.figure(figsize=(6,4))

#criando um gráfico de barras
sns.barplot(x=grouped_df['mes'], y=grouped_df['focuses'], color='blue')

#usando a primeira coluna para o gráfico
with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(grouped_df['mes'], grouped_df['focuses'])
    plt.title('Total Queimadas por Mês', fontsize=15)
    plt.xlabel('Mês')
    plt.tight_layout()
    st.pyplot(fig2)
    

#Terceiro gráfico -----------------------------------------------------------------------

#agrupando os dados pela data e calcule a soma do campo 'focuses'
grouped_df = df.groupby('date')['focuses'].sum().reset_index()

#definindo o estilo Seaborn
sns.set_theme()

#cria um gráfico de linha
fig3, ax3 = plt.subplots(figsize=(10,4))
ax3.plot_date(grouped_df['date'], grouped_df['focuses'], linestyle='solid', color='blue')
ax3.set_title('Histórico Queimadas', fontsize=12)
st.pyplot(fig3)

#texto
st.markdown("Através dos gráficos acima podemos concluir que a região norte é o principal foco do país em relação as queimadas, com exceção do Mato Grosso que aparece com grande representatividade. Outro ponto que podemos concluir é que a sazonalidade das queimadas traz uma dispersão muito grande para os dados, nos períodos de pico próximos a Setembro os casos podem chegar a 30.000, enquanto em períodos como Março os casos ficam abaixo de 2.000. Isso já nos dá alguns indícios de que para comparar de forma mais embasada a gestão dos dois governos no que se diz respeito as queimadas, teremos que aguardar o período de pico deste ano para termos insumos suficientes para fazer a comparação.")


#2º Bloco************************************************************************************************************************
st.subheader("Análises de Variáveis")

#texto
st.markdown("O próximo passo do estudo é avaliar se variáveis como média de temperatura, média de temperatura máxima e precipitação ao longo do mês se relacionam com a quantidade de queimadas.")

# Criando um widget de seleção para selecionar a variável a ser analisada
variaveis = ['Temperatura Máxima', 'Temperatura Média', 'Precipitação']
variavel_selecionada = st.selectbox('Selecione uma variável:', options=list(variaveis))

df_enriquecido = pd.read_csv("dados_queimadas_enriquecidos.csv")
df_enriquecido['date'] = pd.to_datetime(df_enriquecido['date'])

#agrupando os dados pela data e calcule a soma do campo 'focuses'
group_df_variaveis = df_enriquecido.groupby('date').agg({
    'focuses': 'sum', 
    'temperature_2m_max': 'mean', 
    'temperature_2m_mean': 'mean', 
    'precipitation_sum': 'sum'}).reset_index()


#definindo o estilo Seaborn
sns.set_theme()

fig4, ax4 = plt.subplots(figsize=(10,6))
color = 'tab:blue'
ax4.set_ylabel('Queimadas', color=color)
ax4.plot_date(group_df_variaveis['date'], group_df_variaveis['focuses'],linestyle='solid', color='blue')


if variavel_selecionada == "Temperatura Máxima":
    coluna_variavel = "temperature_2m_max"
elif variavel_selecionada == "Temperatura Média":
    coluna_variavel = "temperature_2m_mean"
elif variavel_selecionada == "Precipitação":
    coluna_variavel = "precipitation_sum"  

ax5 = ax4.twinx()  
color = 'tab:red'
ax5.set_ylabel(variavel_selecionada, color=color)  
ax5.plot_date(group_df_variaveis['date'], group_df_variaveis[coluna_variavel],linestyle='solid', color='red')

fig4.tight_layout()  
st.pyplot(fig4)


#texto
st.markdown("A análise de variáveis mostra nitidamente que temos uma grande relação de média de temperatura máxima e média de temperatura ao longo do mês com a quantidade de queimadas, e a variável de precipitação mostrou ter uma relação inversa com a quantidade de queimadas, o que já era esperado. De uma forma geral, pelo que vimos até aqui as variáveis naturais possuem forte correlação com as ocorrências de queimadas, o que sugere que as ações intencionais do homem ou não são tão representativas por não ocorrerem tanto em outros períodos ou que estas ações intencionais ocorrem também em períodos que naturalmente já teríamos mais casos de queimadas devido as altas temperaturas e a a baixa umidade.")

#3º Bloco************************************************************************************************************************
st.subheader("Bolsonaro X Lula")

st.markdown("Como foi dito anteriormente, para avaliar o governo Lula quanto a eficácia no combate as queimadas precisaríamos dar mais tempo, porque os períodos em que as queimadas começam a ser mais frequentes é a partir do meio do ano. Contudo, podemos fazer uma análise preliminar avaliando os dados que temos até o momento de Abril de 2023 e compararmos com os mesmos períodos dos últimos anos do governo Bolsonaro, e por fim utilizaremos o modelo preditivo para projetar Maio.")

#convertendo a coluna 'date' para datetime
df_enriquecido['date'] = pd.to_datetime(df_enriquecido['date'])

#cria as colunas 'year' e 'month'
df_enriquecido['ano'] = df_enriquecido['date'].dt.year
df_enriquecido['mês'] = df_enriquecido['date'].dt.month

#agrupa os dados por ano e mês
comparativo_ano = df_enriquecido.groupby(['ano', 'mês'])['focuses'].sum().reset_index()
comparativo_ano = comparativo_ano[comparativo_ano['ano'] >= 2020]

#cores para os diferentes anos
colors = {2020: 'lightblue', 2021: 'blue', 2022: 'darkblue', 2023: 'red'}

#Gráfico 1
fig6, ax6 = plt.subplots(figsize=(10,6))
for ano in comparativo_ano[(comparativo_ano['ano']>=2020)]['ano'].unique():
    df_ano = comparativo_ano[comparativo_ano['ano'] == ano]
    if ano == 2023:
        df_ano = df_ano[df_ano['mês'] != 5]
    ax6.plot(df_ano['mês'], df_ano['focuses'], color=colors[ano], label=ano)

# Legendas e títulos
ax6.set_xlabel('Mês')
ax6.set_ylabel('Qtd Queimadas')
ax6.legend()
plt.title('Total Queimadas por Mês', fontsize=15)
plt.tight_layout()
st.pyplot(fig6)
    
#Gráfico 2
comparativo_ini_ano = comparativo_ano[comparativo_ano['mês'].isin([1, 2, 3, 4])]

#cria o gráfico de barras
fig7, ax7 = plt.subplots(figsize=(10,6))

#define a largura das barras
bar_width = 0.2

#cria o gráfico para cada ano
for i, ano in enumerate(comparativo_ini_ano['ano'].unique()):
    df_ano_ini = comparativo_ini_ano[comparativo_ini_ano['ano'] == ano]
    #adiciona um valor constante ao argumento `x` do método `bar` para ajustar a posição das barras
    bars = ax7.bar(df_ano_ini['mês'] + i*bar_width, df_ano_ini['focuses'], 
            color=colors[ano], label=ano, width=bar_width)
    
    #adiciona rótulos nas barras
    for bar in bars:
        yval = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
                 ha='center', va='bottom', fontsize=10)

#legendas e títulos
ax7.set_ylabel('Qtd Queimadas')

ax7.legend()

#define os ticks do eixo X para corresponderem ao meio das barras e define os rótulos dos ticks como os meses
ax7.set_xticks([1 + bar_width/2, 2 + bar_width/2, 3 + bar_width/2, 4 + bar_width/2])
ax7.set_xticklabels(['Janeiro', 'Fevereiro', 'Março', 'Abril'])
plt.title('Total Queimadas por Mês', fontsize=15)
plt.tight_layout()
st.pyplot(fig7) 

# Gráfico 3 - Barras de valores acumulados por ano até o mês de abril
#agrupando os dados pelos anos, somando os valores de 'focuses' dos primeiros 4 meses
fig8, ax8 = plt.subplots(figsize=(10,6))
comparativo_acum_ano = comparativo_ano[comparativo_ano['mês'].isin([1, 2, 3, 4])]
comparativo_acum_ano = comparativo_acum_ano.groupby('ano')['focuses'].sum().reset_index()

#define a largura das barras
bar_width = 0.6

#cria o gráfico para cada ano
for i, ano in enumerate(comparativo_acum_ano['ano']):
    #desenha a barra
    bar = ax8.bar(i, comparativo_acum_ano.loc[i, 'focuses'], 
            color=colors[ano], label=ano, width=bar_width)
    
    #adiciona o valor acima da barra
    height = bar[0].get_height()
    ax8.text(bar[0].get_x() + bar[0].get_width()/2,  
            1.01*height,  
            '{}'.format(int(height)),  
            ha='center', va='bottom') 

# Legendas e títulos
ax8.set_xlabel('Ano')
ax8.set_ylabel('Qtd Queimadas')
ax8.legend()

#define os ticks do eixo X para corresponderem ao meio das barras e define os rótulos dos ticks como os anos
ax8.set_xticks(range(len(comparativo_acum_ano)))
ax8.set_xticklabels(comparativo_acum_ano['ano'])

plt.tight_layout()
st.pyplot(fig8)


st.markdown("Até o momento na gestão do governo Lula tivemos menos casos de queimadas acumulados nos 4 primeiros meses do que nos 3 anos anteriores do governo Bolsonaro, mas os meses que serão mais críticos para o governo nesse tema virão agora no meio do ano.")


#4º Bloco************************************************************************************************************************
st.subheader("Modelo preditivo")

st.markdown("Por fim, vamos projetar como será o mês de Maio, já que ainda não temos estes dados, considerando variáveis históricas de queimadas e metereológicas.")

df_projecao = pd.read_csv("dados_queimadas_previsao.csv")

#botão projetar
# if st.button("Projetar"):

#carregamento / instanciamento do modelo pkl
mdl_lgbm = load_model('./pycaret_mdl_xg')

#Acessando o modelo LGBMClassifier
lgbm_model = mdl_lgbm.named_steps['trained_model']

#predict do modelo
ypred = predict_model(mdl_lgbm, data = df_projecao)
df_projecao['focuses'] = ypred['prediction_label']

#Gráfico Projeção
df_projecao_agrupado = df_projecao.groupby(['ano', 'mes'])['focuses'].sum().reset_index()
df_projecao_agrupado = df_projecao_agrupado.rename(columns={'mes':'mês'})
comparativo_ano = comparativo_ano[(comparativo_ano['ano']<2023) | (comparativo_ano['mês']<5)]
# comparativo_ano = comparativo_ano.append(df_projecao_agrupado)
# comparativo_ano = comparativo_ano[comparativo_ano['mês']<6]

# # Cria o gráfico de barras
# fig8, ax8 = plt.subplots(figsize=(10,6))

# # Define a largura das barras
# bar_width = 0.2

# # Cria o gráfico para cada ano
# for i, ano in enumerate(comparativo_ano['ano'].unique()):
#     df_ano_ini = comparativo_ano[comparativo_ano['ano'] == ano]
#     # Adiciona um valor constante ao argumento `x` do método `bar` para ajustar a posição das barras
#     bars = ax8.bar(df_ano_ini['mês'] + i*bar_width, df_ano_ini['focuses'], 
#             color=colors[ano], label=ano, width=bar_width)

#     # Adiciona rótulos nas barras
#     for bar in bars:
#         yval = bar.get_height()
#         ax8.text(bar.get_x() + bar.get_width()/2, yval, int(yval), 
#                  ha='center', va='bottom', fontsize=10)

# # Legendas e títulos
# #ax7.set_xlabel('Mês')
# ax8.set_ylabel('Qtd Queimadas')
# ax8.legend()

# # Define os ticks do eixo X para corresponderem ao meio das barras e define os rótulos dos ticks como os meses
# ax8.set_xticks([1 + bar_width/2, 2 + bar_width/2, 3 + bar_width/2, 4 + bar_width/2, 5 + bar_width/2])
# ax8.set_xticklabels(['Janeiro', 'Fevereiro', 'Março', 'Abril','Maio'])

# plt.tight_layout()
# st.pyplot(fig8)

# st.markdown("Como podemos ver, a projeção indica que iniciaremos o mês de Maio com um pico menor que o ano passado mas maior do que 2020 e 2021. A grande diferença que notamos com estes dois anos é um possível deslocamento do início do pico das queimadas para Maio em 22 e 23, enquanto em 20 e 21, esse período ocorreu de forma mais tardia.")







    

        
