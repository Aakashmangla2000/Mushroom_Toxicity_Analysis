from model import *
import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
import pickle



@st.cache(persist=True)
def load_data():
    data = pd.read_csv("./mushroom.csv")
    label =LabelEncoder()
    for i in data.columns:
        data[i] = label.fit_transform(data[i])
    return data

def main():
    st.title('IS MY MUSHROOM EDIBLE ?')
    st.sidebar.title('MENU')
    st.markdown("## **Introduction**")
    img = Image.open('image1.jpg')
    st.image(img,use_column_width=True)
    st.markdown("**A mushroom or toadstool is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground, on soil, or on its food source.**")
    st.markdown("Mushrooms are unique in the produce section because they are fungi and not vegetables. What we typically think of as a mushroom is the fleshy, fruiting, spore-bearing body of a fungus. The mushrooms we eat are generally composed of a stipe (stem), a pileus (cap), and lamellae (gills). There are, however, many morphological varieties of mushrooms and not all varieties have these features. There are approximately 14,000 different species of mushroom, many of which are inedible.")
    
    
    

    df = load_data()
   
    data = pd.read_csv("./mushroomatt.csv")
    st.markdown("## **About the Data**")
    st.markdown(body="This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and"
        "Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. " 
        "This latter class was combined with the poisonous one. The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; "
         "no rule like ``leaflets three, let it be'' for Poisonous Oak and Ivy.")

    st.sidebar.subheader("ABOUT THE DATA")
    options = st.sidebar.selectbox("Choose an option",('Show Attributes','Show Dataset','Show Both'))
    if options=='Show Attributes':
        st.markdown("### **The Attributes **")
        st.write(data)
    if options=='Show Dataset':
        st.markdown("### **The Dataset**")
        st.write(df) 
    if options=='Show Both':
        st.markdown("### **The Attributes **")
        st.write(data)
        st.markdown("### **The Dataset**")
        st.write(df) 

    st.markdown("## **Structure of Mushrooms**")
    st.sidebar.subheader("FEATURE ANALYSIS")
    st.markdown("### **• Cap of the Mushroon**")
    st.markdown(body="The top part of a mushroom is called the cap. This cap looks similar to an umbrella and acts in a similar way in protection. This protection is most important to the gills and spores that are just below the cap. ")
    
    img = Image.open('cap.jpg')
    st.image(img, use_column_width=True)
    

    if st.sidebar.checkbox("Cap Features",False,key=0):
        st.write('They are differently shaped,varied surface types and colored upper part of the mushroom that protects the gills; it usually resembles a headdress, hence its name.'
        'Through the given graph we will determine the relationship of cap features like shape,surface and colour with the edibility of mushroom in the given data.')

        ##create dimensions
        type_dim = go.parcats.Dimension(
            values = df.type,
            label = "types",
            categoryarray = [0,1],
            categoryorder= 'category ascending',
            ticktext= ['edible','poisonous']

        )
        shape_dim = go.parcats.Dimension(
            values = df['cap-shape'].values,
            label = "cap-shape",
            categoryarray = [0,1,2,3,4,5],
            #categoryorder= 'category ascending',
            ticktext= ['Bell','conical','flat','knobbed','sunken','convex'],

        )
        surface_dim = go.parcats.Dimension(
            values = df['cap-surface'].values,
            label = "cap-surface",
            categoryarray = [0,1,2,3],
            #categoryorder= 'category ascending',
            ticktext= ['fibrous','grooves','smooth','scaly']

        )
        color_dim = go.parcats.Dimension(
            values = df['cap-color'].values,
            label = "cap-color",
            categoryarray = [0,1,2,3,4,5,6,7,8,9],
            #categoryorder= 'category ascending',
            ticktext= ['buff','cinnamon','red','gray','brown','pink','green','purple','white','yellow']

        )
        color = df.type
        colorscale = [[0, 'darkturquoise'], [1, 'steelblue']]
        #colorscale = [[0, 'lightblue'], [1, 'violet']]
        fig = go.Figure(data = [go.Parcats(dimensions=[type_dim, shape_dim, surface_dim,color_dim],
        line={'color': color, 'colorscale': colorscale},
        hoveron='dimension', hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='fixed')])
        fig.update_layout(height=500, width=800,paper_bgcolor='rgb(243, 243, 243)')
        st.write(fig)
        st.write('We have used parallel category diagram as parallel plot or parallel coordinates plot allows to compare the feature of several individual observations (series) on a set of numeric variables. Each vertical bar represents a variable and often has its own scale. (The units can even be different). Values are then plotted as series of lines connected across each axis.')
        st.write('From the graph we can infer that:  \n ➡ Maximum number of the mushrooms whether posionous or edible have their cap-shape as either convex or sunken.\n'
        ' \n  ➡ Maximum number of the mushrooms have their surfaces as smooth, grooves and scaly. \n '
        '\n  ➡ Maximum number of the mushrooms have their colour has green.')

    st.markdown("### **• Bruises of the Mushroon**")
    st.markdown(body="Mushroom bruising involves nicking the top and bottom of the mushroom cap and observing any colour changes. As specimens that are not fresh don’t give reliable results, it is important to do this within the first 30 minutes of picking the mushroom.")

    if st.sidebar.checkbox("Bruises",False,key=1):
        st.write('The following Bargraph will help you understand the relationship of mushroom bruising with edibility.')
        # a=df.groupby(['bruises','type']).count()
        # st.write(a)
        # b=df['bruises'].where(df['type']==0)  
        # st.write(b.count())
        data_bar=[
        go.Histogram(
            histfunc="count",
            x= df['bruises'].where(df['type']==0),  
            name = "edible",
            marker=dict(color="coral",line=dict(color='red', width=4)),
            opacity=0.75

        ),
        go.Histogram(
            histfunc = "count",
            x = df['bruises'].where(df['type']==1),
            name = "poisonous",
            marker=dict(color="hotpink",line=dict(color='purple', width=4)),
            opacity=0.75
               )

        ]
        layout = go.Layout(
            title='Bruises analysis and Edibility',
            xaxis=dict(
                title='Feature Value',
                
                ),
            yaxis=dict(
                    title='Count'
                ),
            bargap=0.3,
            bargroupgap=0.2, paper_bgcolor='mistyrose',
            plot_bgcolor="mistyrose")
        fig = go.Figure(data=data_bar, layout=layout)
        st.write(fig)
        st.write('From the graph we can infer that:  \n ➡ Mushrooms with no bruises are more prominent.\n'
        ' \n  ➡ Poisonous mushrooms have the least number of bruises \n '
        )

    st.markdown("### **• Odor of the Mushrooms**")
    st.markdown(body="A odor of a mushroom is almond or anise, it is edible or if it is musty, creosote, pungent, fishy, spicy or foul, it is poisonous. Moreover, even most of the mushrooms which have none odor are poisonous. Odor feature is a really important feature to distinguish between a mushroom that is edible or poisonous.")

    if st.sidebar.checkbox("Odor",False,key=2):
        st.write('The following Bargraph will help you understand the relationship of mushroom bruising with edibility.')
        odor_edible=df['odor'].where(df['type']==0).value_counts()
        odor_poisonous=df['odor'].where(df['type']==1).value_counts()
        #st.write(odor_edible)
        # st.write(odor_poisonous.index)
        
        layout = go.Layout(
            yaxis = go.layout.YAxis(
                title='odor',
                tickvals=[5.0, 3.0, 0.0, 2.0, 7.0, 8.0, 6.0, 1.0, 4.0],
                ticktext=['none', 'anise', 'almond', 'foul', 'spicy','fishy', 'pungent', 'creosote', 'musty']
                ),
            xaxis = go.layout.XAxis(
                range=[-2000,2000],
                tickvals=[-2000, -1600, -1200, -800, -400, 0, 400, 800, 1200,1600,2000],
                ticktext=[2000, 1600, 1200, 800, 400, 0, 400, 800, 1200,1600,2000],
                title='count'
            ),
            barmode='overlay',
            bargap=0.2,
            paper_bgcolor="linen",
            #paper_bgcolor='rgb(243, 243, 243)',
            plot_bgcolor='linen'

        )
        data_graph = [go.Bar(y=odor_edible.index,
               x=odor_edible,
               orientation='h',
               name='Edible',
               hoverinfo='text',
               text=odor_edible.astype('int'),
               marker=dict(color='navy'),
               width=0.8
               ),
               go.Bar(y=odor_poisonous.index,
               x=-1*odor_poisonous,
               orientation='h',
               name='Poisonous',
               hoverinfo='text',
               text=odor_poisonous.astype('int'),
               marker=dict(color='sandybrown'),
               width=0.8
               )]
        fig = go.Figure(data=data_graph, layout=layout)
        st.write(fig)
        st.write('From the graph we can infer that:  \n ➡ Mostly edible mushrooms have no order.\n'
        ' \n ➡ Poisonous mushrooms show a variety of odors with maximum count seen in foul category. \n '
        
        )
        
    st.markdown("### **• The Gill Properties of mushroom**")
    st.markdown(body="The gills of mushroom produce spores which help in the process of reproduction. The gills of mushroom hang vertically below its cap and allow the spores to fall in the downward direction into the air. These spores are further carried out by natural pollinators such as wind, insects and animals for the process of reproduction.")
    img = Image.open('gill3.jpg')
    st.image(img,use_column_width=True)    
    if st.sidebar.checkbox("Gill Properties",False,key=2):
        st.write('Examining the gills is important when identifying mushrooms. Mycologists have many terms to describe gill structure,'
                 'some very precise and complicated. Here are some of the more common features of mushroom gills:\n' 
                 '\n ➡ Stem attachment: Observe how the gills attach to the stem. They may be "free", meaning they do not attach to the stem at all as with portobellos or amanitas. Or they may be attached directly or by a notch.\n '
                 '\n ➡ Color and bruising: Note the color of your mushrooms gills. Sometimes they will be very different from the cap color. If you apply pressure to them with your fingernail or a knife they may bruise a different color. These features can help you identify mushrooms with gills.\n'
                 '\n ➡ Gill spacing: Notice how many gills are packed into the underside of the cap. Is it crowded with many gills in one place or is there space between them? This can admittedly be hard to judge!')
        
        
        ga=df.groupby(['type','gill-attachment']).count()
        gs=df.groupby(['type','gill-spacing']).count()
        gsi=df.groupby(['type','gill-size']).count()
        gc=df.groupby(['type','gill-color']).count()
        st.write(gc)
        
        
        # b=df['bruises'].where(df['type']==0)  
        # st.write(b.count())
        
        
        trace0 = go.Scatter(
            x = [0,1],
            y = ga.iloc[0:2,0],
            mode = 'markers',
            name = 'Edible',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "navy",
                            opacity= 0.7
                        )
        )
        trace1 = go.Scatter(
            x = [0,1],
            y = ga.iloc[2:4,0],
            mode = 'markers',
            name = 'Poisonous',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "lime",
                            opacity= 0.7,
                        symbol=220
                        )
        )
        trace2 = go.Scatter(
            x = [0,1],
            y = gs.iloc[0:2,0],
            mode = 'markers',
            name = 'Edible',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "navy",
                            opacity= 0.7
                        )
        )
        trace3 = go.Scatter(
            x =[0,1],
            y = gs.iloc[2:4,0],
            mode = 'markers',
            name = 'Poisonous',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "lime",
                            opacity= 0.7,
                            symbol=220
                        )
        )
        trace4 = go.Scatter(
            x = [0,1],
            y = gsi.iloc[0:2,0],
            mode = 'markers',
            name = 'Edible',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "navy",
                            opacity= 0.7
                        )
        )
        trace5 = go.Scatter(
            x = [0,1],
            y = gsi.iloc[2:4,0],
            mode = 'markers',
            name = 'Poisonous',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "lime",
                            opacity= 0.7,
                        symbol=220
                        )
        )
        trace6 = go.Scatter(
            x = [1,2,3,4,5,6,7,9,10,11],
            y = gc.iloc[0:10,0],
            mode = 'markers',
            name = 'Edible',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "navy",
                            opacity= 0.7
                        )
        )
        trace7 = go.Scatter(
            x = [0,2,3,4,5,6,7,8,9,10,11],
            y = gc.iloc[10:20,0],
            mode = 'markers',
            name = 'Poisonous',
            marker= dict(size= 14,
                            line= dict(width=1),
                            color= "lime",
                            opacity= 0.7,
                        symbol=220
                        )
        )

        fig = tools.make_subplots(rows=2, cols=2, 
                                subplot_titles=('Gill Attachment','Gill Size', 'Gill Spacing',"Gill Color"))


        fig.append_trace(trace0, 1, 1)
        fig.append_trace(trace1, 1, 1)
        fig.append_trace(trace2, 2, 1)
        fig.append_trace(trace3, 2, 1)
        fig.append_trace(trace4, 1, 2)
        fig.append_trace(trace5, 1, 2)
        fig.append_trace(trace6, 2, 2)
        fig.append_trace(trace7, 2, 2)

        fig['layout'].update(showlegend=False,height=800, width=800, title='Gill Properties' ,paper_bgcolor='rgb(243, 243, 243)',
            plot_bgcolor="linen")
        st.write(fig)
        st.write('From the above scatter plot we can observe that: \n'
        '\n ➡ Gill attachment and gill spacing are not distinctive features.\n'
        '\n ➡ Most of the mushrooms in dataset have free gill attachment and close gill spacing.\n'
        '\n ➡ Gill- size and gill colors can be. We can say a mushroom which has buff and green gill color is poisonous or red and orange gill color is edible. \n'
        '\n ➡ Most of the edible mushrooms have broad gill-size but numbers of poisonous mushrooms which have broad or narrow are close to each other. \n'
        
        )

        
    st.markdown("### **• The Stalk Properties of mushroom**")
    st.markdown(body="The gills of mushroom produce spores which help in the process of reproduction. The gills of mushroom hang vertically below its cap and allow the spores to fall in the downward direction into the air. These spores are further carried out by natural pollinators such as wind, insects and animals for the process of reproduction.")
    if st.sidebar.checkbox("Stalk Properties",False,key=3):
        # gill_att = df.groupby(['type','gill-attachment']).count()
        # gill_att1 = df.groupby(['type','gill-size']).count()
        # gill_att2 = df.groupby(['type','gill-spacing']).count()
        # gill_att3 = df.groupby(['type','gill-color']).count()
       
        # st.write(gill_att)
        # st.write(gill_att1)
        # st.write(gill_att2)
        # st.write(gill_att3)
        
        # fig.update_layout(margin = dict(t=10, l=10, r=10, b=10))
        # st.write(fig)
        df1 = pd.read_csv('./gills.csv')
        
        fig0 = px.sunburst(df1,path=['stalk-shape','type'])
        fig1 = px.sunburst(df1,path=['stalk-root','type'])
        fig2 = px.sunburst(df1,path=['stalk-surface-above-ring','type'])
        fig3 = px.sunburst(df1,path=['stalk-surface-below-ring','type'])
        fig = go.Figure()
        fig.add_trace(
            go.Sunburst(
            labels=fig0['data'][0]['labels'].tolist(),
            parents=fig0['data'][0]['parents'].tolist(),
            values=fig0['data'][0]['values'].tolist(),
            ids=fig0['data'][0]['ids'].tolist(),
            domain=dict(column=0,row=0),
            hovertemplate='<b>%{label} </b> <br> mushroom_count: %{value}<br> type: %{parent}',
            branchvalues='total',
            marker = {
            "line": {"width":3},
            "colors": ['#DEB887','#DAA520','#800080','#DB7093','#00BFFF','#7B68EE','#00FF7F','#800000']

            },
            insidetextorientation='radial',
            name='stalk-shape'
            
            
                            ))
            
        fig.add_trace(
            go.Sunburst(
            labels=fig1['data'][0]['labels'].tolist(),
            parents=fig1['data'][0]['parents'].tolist(),
            values=fig1['data'][0]['values'].tolist(),
            ids=fig1['data'][0]['ids'].tolist(),
            domain=dict(column=1,row=0),
            branchvalues='total',
            hovertemplate='<b>%{label} </b> <br> mushroom_count: %{value}<br> type: %{parent}',
            
            marker = {
            "line": {"width":3},
            "colors": ['#DEB887','#DAA520','#800080','#DB7093','#00BFFF','#7B68EE','#00FF7F','#800000']

            },
            insidetextorientation='radial',
            name='stalk-root'
                            ))
            
        fig.add_trace(
            go.Sunburst(
            labels=fig2['data'][0]['labels'].tolist(),
            parents=fig2['data'][0]['parents'].tolist(),
            values=fig2['data'][0]['values'].tolist(),
            ids=fig2['data'][0]['ids'].tolist(),
            domain=dict(column=0,row=1),
            branchvalues='total',
            hovertemplate='<b>%{label} </b> <br> mushroom_count: %{value}<br> type: %{parent}',
            
            marker = {
            "line": {"width":3},
            "colors": ['#DEB887','#DAA520','#800080','#DB7093','#00BFFF','#7B68EE','#00FF7F','#800000']

            },
            insidetextorientation='radial',
            name='stalk-surface-above-ring'
                            ))
            
        fig.add_trace(
            go.Sunburst(
            labels=fig3['data'][0]['labels'].tolist(),
            parents=fig3['data'][0]['parents'].tolist(),
            values=fig3['data'][0]['values'].tolist(),
            ids=fig3['data'][0]['ids'].tolist(),
            domain=dict(column=1,row=1),
            branchvalues='total',
            hovertemplate='<b>%{label} </b> <br> mushroom_count: %{value}<br> type: %{parent}',
            
            marker = {
            "line": {"width":3},
            "colors": ['#DEB887','#DAA520','#800080','#DB7093','#00BFFF','#7B68EE','#00FF7F','#800000','#FFE4E1','#191970','#556B2F']

            },
            insidetextorientation='radial',
            name='stalk-surface-below-ring'
                            )
            
                )
        # fig.update_layout(
        #     grid= dict(columns=2, rows=2),
        #     margin = dict(t=0, l=0, r=0, b=0),
        #     uniformtext=dict(minsize=10, mode='hide')

        #     )
        fig['layout'].update(
            showlegend=True, 
            paper_bgcolor='ivory',
            plot_bgcolor="ivory", 
            grid= dict(columns=2, rows=2),
            margin = dict(t=0, l=0, r=0, b=0),
            uniformtext=dict(minsize=10, mode='hide'),
            title={
            'text': "Stalk properties",
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'})

        st.write(fig)

        value1=df.groupby(['type','stalk-color-above-ring']).count()
        #st.write(value1.iloc[0:7,0].index)
        # label1=stalk_color_above_ring_edible.index
        value2=df.groupby(['type','stalk-color-below-ring']).count()
        #st.write(value2)
        
        

        trace1=go.Bar(
            x = value1.iloc[0:7,0],
            y = value1.iloc[0:7,0].index.get_level_values('stalk-color-above-ring'),
            name='Edible- Stalk Color Above Ring',
            orientation = 'h',
            marker = dict(
                color = "darksalmon",
                line = dict(
                    color = 'rgba(58, 71, 80, 1.0)',
                    width = 2),
                opacity=0.8,
            ))
            
        trace2=go.Bar(
            x = value1.iloc[7:13,0],
            y = value1.iloc[7:13,0].index.get_level_values('stalk-color-above-ring'),
            name='Poisonous-Stalk Color Above Ring',
            orientation = 'h',
            marker = dict(
                color = "plum",
                line = dict(
                    color = 'rgba(58, 71, 80, 1.0)',
                    width = 2), opacity=0.8))
            
        trace3=go.Bar(
            x = value2.iloc[0:7,0],
            y = value2.iloc[0:7,0].index.get_level_values('stalk-color-below-ring'),
                    name='Edible-Stalk Color Below Ring',
            orientation = 'h',
            marker = dict(
                color = "palegreen",
                line = dict(
                    color = 'rgba(58, 71, 80, 1.0)',
                    width = 2), opacity=0.8))
        
        trace4=go.Bar(
            x =value2.iloc[7:13,0] ,
            y =value2.iloc[7:13,0].index.get_level_values('stalk-color-below-ring'),
                    name='Poisonous- Stalk Color Below Ring',
            orientation = 'h',
            marker = dict(
                color = "sienna",
                line = dict(
                    color = 'rgba(58, 71, 80, 1.0)',
                    width = 2), opacity=0.8))
        

        fig1= tools.make_subplots(rows=1, cols=2,subplot_titles=('Stalk Color Counts Above Ring','Stalk Color Counts Below Rings'))

        fig1.append_trace(trace1, 1, 1)
        fig1.append_trace(trace2, 1, 1)
        fig1.append_trace(trace3, 1, 2)
        fig1.append_trace(trace4, 1, 2)

        fig1['layout'].update(
            showlegend=True,
            height=600, width=800, 
            barmode='stack',
            legend=dict(x=.58, y=-0.1,orientation="h",font=dict(size=11,color='#000')),
            title='Stalk Colors Above and Below Ring',
            yaxis = go.layout.YAxis(
                title='colour',
                tickvals=[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0],
                ticktext=['buff','cinnamon','red','grey', 'brown','orange', 'pink', 'white','yellow']
                ))
        st.write(fig1)
        st.write('From the above plots we can observe that: \n'
        '\n ➡ The rooted stalk-root is seen in edible mushrooms whereas club root type is seen in edible mushroom largely as compared to posionous.\n'
        '\n ➡ The stalk surface above ring type silky is mostly seen poisonous. \n'
        '\n ➡ The stalk surface below ring types scaly and fibrous is mostly seen edible.  \n'
        
        
        )
    st.markdown("### **• The Veil Properties of mushroom**")
    st.markdown(body="A veil or velum, in mycology, is one of several structures in fungi, especially the thin membrane that covers the cap and stalk of an immature mushroom.")
    if st.sidebar.checkbox("Veil Properties",False,key=4):
        st.markdown("Following bar chart shows number of veil colors according to mushroom classes. Most of the mushrooms have white veil color in dataset. Also edible mushrooms veil colors can be orange or brown.")
        value1=df.groupby(['type','veil-color']).count()
       




        trace1 = go.Bar(
            x=value1.iloc[0:3,0].index.get_level_values('veil-color'),
            y=value1.iloc[0:3,0],
            text=['brown','orange'],
            textposition = 'auto',
            name="Edible",
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
                ),
            opacity=0.6
        )

        trace2 = go.Bar(
            x=value1.iloc[3:5,0].index.get_level_values('veil-color'),
            y=value1.iloc[3:5,0],
            text=['white','yellow'],
            name="Poisonous",
            textposition = 'auto',
            marker=dict(
                color='rgb(58,200,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
                ),
            opacity=0.6
            
        )
        Layout=go.Layout(
            title='Veil Colors',
            barmode='stack',
            paper_bgcolor='rgba(245, 246, 249, 1)',
            plot_bgcolor='rgba(245, 246, 249, 1)'
        
        )

        data65 = [trace1,trace2]
        fig = go.Figure(data=data65, layout=Layout)
        st.write(fig)
        st.write('From the above plot we can observe that: \n'
        '\n ➡ Both the categories of mushrooms have white as the most predominant veil colour. \n'
        )
                
    st.markdown("### **• The Spore Print Colour Properties of mushroom**")
    st.markdown(body="The spore print is the powdery deposit obtained by allowing spores of a fungal fruit body to fall onto a surface underneath. It is an important diagnostic character in most handbooks for identifying mushrooms. ")
    if st.sidebar.checkbox("Spore Print Properties",False,key=5):
        st.markdown("Following Pie chart shows spore print colours according to mushroom classes.")    
        value1=df.groupby(['type','spore-print-color']).count() 
        #st.write(value1)
        
        fig = {
            "data": [
                {
                "values": [1744,1648,576,48,48,48,48,48],
                "labels": ['Brown','Black','White','Orange',"Purple","Chocolate","Yellow","Buff"],
                "domain": {"column": 0},
                "name": "Edible Mushrooms",
                "hoverinfo":"label+percent+name",
                "type": "pie",
                    "hole": .4,
                    'marker': {'colors': ['brown', 'black', 'white', 'orange',"purple","sienna","yellow","peru"],
                            "line":{"color":'#000000',"width":2}}
                },
                {
                "values": [1812,1584,224,224,72],
                "labels": ["White","Chocolate","Brown","Black","Green"],   
                "domain": {"column": 1},
                "name": "Poisonous Mushrooms",
                "hoverinfo":"label+percent+name",
                    "hole": .4,
                "type": "pie",
                    "marker": {"colors":["white","sienna","brown","black","green"],
                            "line":{"color":'#000000',"width":2}}

                }],
            "layout": {
                
                    "title":"Edible and Poisonous Mushrooms Spore Print Color Percentages",
                    "grid": {"rows": 1, "columns": 2},
                    "annotations": [
                        {
                            "font": {
                                "size": 20
                            },
                            "showarrow": False,
                            "text": "Edible",
                            "x": 0.20,
                            "y": 1.05
                        },
                        {
                            "font": {
                                "size": 20
                            },
                            "showarrow": False,
                            "text": "Poisonous",
                            "x": 0.85,
                            "y": 1.05
                        }
                    ]
                }
            } 
        fig = go.Figure(data=fig['data'],layout=fig['layout'])
        
        st.write(fig) 
        st.write('From the above plot we can observe that: \n'
        '\n ➡ Most of the edible mushrooms are either brown or black in colour. \n'
        '\n ➡ Most of the poisonous mushrooms are either white or Chocolate in colour. \n'
        )  

    st.markdown("### **• Population & Habitat of the Mushroon**")
    st.markdown(body="Mushrooms are found in a great variety of habitats, although each species may limited in the number of these it can occupy. For example, a mushroom that usually grows on rotting logs in the forest is unlikely to be found in sand dunes. This is one of the great appeals of mushroom hunting: visit a habitat new to you and you will probably find species of mushrooms you have never seen.")

    if st.sidebar.checkbox("Population & Habitat",False,key=1):
        st.write('The following Parallel plot will help you understand the relationship of Population and Habitat.')
         ##create dimensions
        type_dim = go.parcats.Dimension(
            values = df.type,
            label = "types",
            categoryarray = [0,1],
            # categoryorder= 'category ascending',
            ticktext= ['edible','poisonous']

        )
        population_dim = go.parcats.Dimension(
            values = df['population'].values,
            label = "Population",
            categoryarray = [0,1,2,3,4,5,6],
            # categoryorder= 'category ascending',
            ticktext= ['abundant','clustered','numerous','scattered','several','solitary'],

        )
        habitat_dim = go.parcats.Dimension(
            values = df['habitat'].values,
            label = "Habitat",
            categoryarray = [0,1,2,3,4,5,6],
            # categoryorder= 'category ascending',
            ticktext= ['woods','grasses','leaves','meadows','paths','urban','waste']

        )
        
        color = df.type
        colorscale = [[0, 'rebeccapurple'], [1, 'lightsalmon']]
        #colorscale = [[0, 'lightblue'], [1, 'violet']]
        fig = go.Figure(data = [go.Parcats(dimensions=[type_dim, population_dim, habitat_dim ],
        line={'color': color, 'colorscale': colorscale},
        hoveron='dimension', hoverinfo='count+probability',
        labelfont={'size': 18, 'family': 'Times'},
        tickfont={'size': 16, 'family': 'Times'},
        arrangement='fixed')])
        fig.update_layout(height=500, width=800,paper_bgcolor='rgb(243, 243, 243)')
        st.write(fig)
        st.write('We have used parallel category diagram as parallel plot or parallel coordinates plot allows to compare the feature of several individual observations (series) on a set of numeric variables. Each vertical bar represents a variable and often has its own scale. (The units can even be different). Values are then plotted as series of lines connected across each axis.')
        st.write('From the graph we can infer that:  \n ➡ Maximum number of the posionous mushrooms have a population type of either several or solitary.\n'
        ' \n  ➡ Edible mushrooms show diversity in their population type.\n '
        '\n  ➡ Maximum number of edible mushrooms woods or grasses type habitat.')

    main_mod()

    # @st.cache(persist=True)
    # def split(df):
    #     X = df.drop(columns=['type'])
    #     Y = df.type
    #     X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
    #     return X_train,X_test,Y_train,Y_test

    # X_train,X_test,Y_train,Y_test = split(df)

    
     

    # loaded_model_svc = pickle.load(open('svc_model.sav', 'rb'))
    # result = loaded_model_svc.score(X_test, Y_test)
    # st.write(result)
    
    # loaded_model_rf = pickle.load(open('ran_fo_model.sav', 'rb'))
    # result1 = loaded_model_rf.score(X_test, Y_test)
    # st.write(result1)

    # loaded_model_lr = pickle.load(open('lr_model.sav', 'rb'))
    # result2 = loaded_model_lr.score(X_test, Y_test)
    # st.write(result2)
    # y_pred = loaded_model_svc.predict(X_test)
    # cm = confusion_matrix(Y_test, y_pred)
    # # confusion_matrix = confusion_matrix.astype(int)

    # layout = {
    #     "title": "Confusion Matrix", 
    #     "xaxis": {"title": "Predicted value"}, 
    #     "yaxis": {"title": "Real value"}
    # }

    # fig = go.Figure(data=go.Heatmap(z=cm,x=['edible','poisonous'],y=['edible','poisonous']),
    #                 layout=layout)
    # st.write(fig)


    
    
   
        
    
          

    
    
if __name__ == "__main__":
    main()    







   










