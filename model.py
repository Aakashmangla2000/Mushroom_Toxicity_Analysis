from datacleaning import *
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
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score





@st.cache(persist=True)

def load_data():
    data = pd.read_csv("./mushroom.csv")
    label =LabelEncoder()
    for i in data.columns:
        data[i] = label.fit_transform(data[i])
    return data

df = load_data()

def main_mod():
    st.markdown("## **The Models** ")
    st.markdown(body="Now in this section I have applied the most prominent algorithms of machine learning- Lositic Regression, Random Forest and Support vector machine."
    "Here, we will discover the most appropriate model for our dataset.")
    
    @st.cache(persist=True)
    def split(df):
        X = df.drop(columns=['type','cap-shape','cap-color','gill-attachment','stalk-shape','veil-type', 'veil-color', 'ring-number'])
        Y = df.type
        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
        return X_train,X_test,Y_train,Y_test,X,Y
    
    X_train,X_test,Y_train,Y_test,X,Y= split(df)
    # print(X_train)
    # print(X_test)
    loaded_model_svc = pickle.load(open('svc_model.sav', 'rb'))
    y_pred_svc = loaded_model_svc.predict(X_test)
    
    
       
        
    loaded_model_rf = pickle.load(open('ran_fo_model.sav', 'rb'))
    y_pred_rf = loaded_model_rf.predict(X_test)
    
    

    loaded_model_lr = pickle.load(open('lr_model.sav', 'rb'))
    y_pred_lr = loaded_model_lr.predict(X_test)
    

    def cross():
        st.markdown("### **Cross-Validation Score **")
         
        lr_score = np.mean(cross_val_score(loaded_model_lr, X, Y, scoring='accuracy', cv = 5))
        rf_score = np.mean(cross_val_score(loaded_model_rf, X, Y, scoring='accuracy', cv = 5))
        svc_score = np.mean(cross_val_score(loaded_model_svc, X, Y, scoring='accuracy', cv = 5))
        cross_val = [svc_score,rf_score,lr_score]
        cross_val_new = pd.DataFrame(cross_val,index=['Support Vector Machine','Random Forest','Logistic Regression'])
        return(cross_val_new)

    b = [y_pred_lr,y_pred_rf,y_pred_svc] 
    d =pd.DataFrame(b,index=['Logistic Regression','Random Forest','Support Vector Machine'])
      
    def conf_matrix():
        cm_svc = confusion_matrix(Y_test, y_pred_svc)
        cm_rf = confusion_matrix(Y_test, y_pred_rf)
        cm_lr = confusion_matrix(Y_test, y_pred_lr)
        
        trace1 = {
            "type": "heatmap",
            "z": cm_svc,
            "x": ['edible','poisonous'],
            "y": ['edible','poisonous'],
            
            "xaxis": "x1",
            "yaxis": "y1",
            "colorbar": {
                "len":1,
                "title":"svm"
            },
            "showscale": True,
            "colorscale": "YlGnBu"

        }
        trace2 = {
            "type": "heatmap",
            "z": cm_rf,
            "x": ['edible','poisonous'],
            "y": ['edible','poisonous'],
            "xaxis": "x2",
            "yaxis": "y2",
            "colorbar": {
                "len":1,
                "title":"Rf"
            },
            "showscale": False,
            "colorscale": "YlGnBu"

        }
        trace3 = {
            "type": "heatmap",
            "z": cm_lr,
            "x": ['edible','poisonous'],
            "y": ['edible','poisonous'],
            "xaxis": "x3",
            "yaxis": "y3",
            "colorbar": {
                "len":1,
                "title":"Lr"
            },
            "showscale": False,
            "colorscale": "YlGnBu"

        }
        data = [trace1,trace2,trace3]
        layout ={
            "title": "Model's Confusion Matrix",
            "width": 900,
            "height": 400,
            "xaxis1": {
                "title": "Support Vector Classifier",
                "anchor": "y1",
                "domain": [0.0,0.2125]
            },
            "xaxis2": {
                "title": "Random Forest Classifier",
                "anchor": "y2",
                "domain": [0.2625,0.475]
            },
            "xaxis3": {
                "title": "Logestic Regression Classifier",
                "anchor": "y3",
                "domain": [0.525,0.7375]
            },
            "paper_bgcolor": 'rgba(245, 246, 249, 1)',
            "plot_bgcolor": 'RGB(255, 245, 238)'
            

        }
        fig = go.Figure(data=data, layout=layout)
        return(fig)
        

    def ROC_curve():
        fig = go.Figure()
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

    
        for label, row in d.iterrows():
            fpr, tpr, thresholds = roc_curve(Y_test, row)
            name = f'{label} (AUC={auc(fpr, tpr):.4f})'
            fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=name))
        
        fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
        )
        return(fig)




    st.sidebar.subheader("MODEL ANALYSIS")
    options = st.sidebar.selectbox("Choose an option",('Confusion Matrix','Cross-Val Score','ROC-Curve','Show Graphs'))
    

    if options=='Confusion Matrix':
        st.markdown("### **Confusion Matrix **")
        fig1 = conf_matrix()
        st.write(fig1)
    if options=='Cross-Val Score':
        table=cross()
        st.write(table)
    if options=='ROC-Curve':
        st.markdown("### **ROC curve**")
        fig1 = ROC_curve()
        st.write(fig1) 
    if options=='Show Graphs':
        st.markdown("### **Confusion Matrix **")
        fig1 = conf_matrix()
        st.write(fig1)
        st.markdown("### **ROC curve**")
        fig1 = ROC_curve()
        st.write(fig1) 
    

    st.markdown("## **The Prediction** ")
    st.markdown(body="Finally we are at the predictor put the details of your mushroom and analyze its edibility.")
    print(df.columns)
    st.markdown("### **Predictor Form**")
    
    st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    options1 = "bell"
    # ,"conical","convex","flat", "knobbed","sunken")
    # cap_shape= st.empty()
    #value1 = cap_shape.radio("cap-shape", options1,key=0)
    #value1 = options1
    
    

    options2 = ("fibrous","grooves","scaly","smooth")
    cap_surface= st.empty()
    value2 = cap_surface.radio("cap-surface", options2,key=1)
    

    options3 = "brown"
    # ,"buff","cinnamon","gray","green","pink","purple","red","white","yellow")
    # cap_color= st.empty()
    # value3 = cap_color.radio("cap-color", options3,key=2)
    #value3 = options3
    
    

    options4 =  ("true","false")
    bruises= st.empty()
    value4 = bruises.radio("bruise", options4,key=3)
    

    options5 =  ("almond","anise","creosote","fishy","foul", "musty","none","pungent","spicy")
    odor= st.empty()
    value5 = odor.radio("odor", options5,key=4)
   

    options6 =  "attached"
    # ,"descending","free","notched")
    # gill_attach= st.empty()
    # value6 = gill_attach.radio("gill-attachment", options6,key=5)
    #value6 = options6
    

    options7 =  ("close","crowded","distant")
    gill_spacing= st.empty()
    value7 = gill_spacing.radio("gill-spacing", options7,key=6)
    

    options8 =  ("broad","narrow")
    gill_size= st.empty()
    value8 = gill_size.radio("gill-size", options8,key=7)
    
    
    options9 =  ("black","brown","buff","chocolate","gray","green","orange","pink","purple","red", "white","yellow")
    gill_color= st.empty()
    value9 = gill_color.radio("gill-color", options9,key=8)
    

     
    options10 =  "enlarging"
    # ,"tapering")
    # stalk_shape= st.empty()
    # value10 = stalk_shape.radio("stalk-shape", options10,key=9)
    #value10 = options10
    

    options11 =  ("bulbous","club","cup","equal","rhizomorphs","rooted","missing")
    stalk_root= st.empty()
    value11 = stalk_root.radio("stalk-root", options11,key=10)
    

    options12 =  ("fibrous","scaly","silky","smooth")
    sur_above_ring= st.empty()
    value12 = sur_above_ring.radio("stalk-surface-above-ring", options12,key=11)
    

    options13 =  ("fibrous","scaly","silky","smooth")
    sur_below_ring= st.empty()
    value13 = sur_below_ring.radio("stalk-surface-below-ring", options13,key=12)
    

    options14 =  ("brown","buff","cinnamon","gray","orange","pink","red","white","yellow")
    col_above_ring= st.empty()
    value14 = col_above_ring.radio("stalk-surface-below-ring", options14,key=13)
   

    options15 =  ("brown","buff","cinnamon","gray","orange","pink","red","white","yellow")
    col_below_ring= st.empty()
    value15 = col_below_ring.radio("stalk-color-below-ring", options15,key=14)
    

    options16 =  "partial"
    # ,"universal")
    # veil_type= st.empty()
    # value16 = veil_type.radio("veil-type", options16,key=15)
    #value16 = options16
    

    options17 =  "brown"
    # ,"orange","white","yellow")
    # veil_color= st.empty()
    # value17 = veil_color.radio("veil-color", options17,key=16)
    #value17 = options17
    

    #value18 = "one"
   
    options19 = ("cobwebby","evanescent","flaring","large","none","pendant","sheathing","zone")
    ring_type = st.empty()
    value19 = ring_type.radio("ring-type", options19,key=0)
    

    options20 =  ("black","brown","buff","chocolate","green","orange","purple","white","yellow")
    spore_color= st.empty()
    value20 = spore_color.radio("spore-print-color", options20,key=17)
    

    options21 =  ("abundant","clustered","numerous", "scattered","several","solitary")
    population= st.empty()
    value21 = population.radio("population", options21,key=19)
    

    options22 =  ("grasses","leaves","meadows","paths","urban","waste","woods")
    habitat= st.empty()
    value22 = habitat.radio("habitat", options22,key=20)
    

    pred_data= np.array([value2,value4,value5,value7,value8,value9,
               value11, value12,value13,value14, value15,value19,value20,value21,value22])
    
    pred_data =pd.DataFrame(pred_data.reshape(-1, len(pred_data)),columns=X_train.columns)
    data=pred_data
    new_data = clean_main(data)
    #print(new_data.head())
    a =loaded_model_rf.predict(new_data)
    
    if a[0]==0:
        st.write(a[0])
        st.markdown("The mushroom is edible")
    
    if a[0]==1:
        st.write(a[0])
        st.markdown("The mushroom is Not edible")
    
    print(X.columns)







    
   
    

   



    
    


    
    
   
        
    
          














if __name__ == "__main_mod__":
    main_mod() 