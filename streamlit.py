import pandas as pd
import streamlit as st 
# import numpy as np

from sqlalchemy import create_engine
import pickle, joblib

imp_enc_scale = joblib.load('imp_enc_scale')
winsor = joblib.load('winsor')
boost_model = pickle.load(open('Gradiantboosting.pkl', 'rb'))


def predict_machine_status(data):

    
    data.drop(['Date'],axis = 1, inplace  = True)            
    clean1 = pd.DataFrame(imp_enc_scale.transform(data), columns = imp_enc_scale.get_feature_names_out())   
    clean1[list(clean1.iloc[:,0:-2].columns)] = winsor.transform(clean1[list(clean1.iloc[:,0:-2].columns)])

   
    prediction = pd.DataFrame(boost_model.predict(clean1), columns = ['Pred_Machine_status'])
    
    final = pd.concat([prediction, data], axis = 1)

    return final



def main():
    st.title("Machine Status prediction")
    st.sidebar.title("Fuel Pump Machine prediction")
    
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;"> Fuel Pump machine Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html = True)
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:blue;padding:10px">
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
   
    result = ""
    
    if st.button("Predict"):
        result = predict_machine_status(data)
    st.dataframe(result) 
    st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        

if __name__=='__main__':
    main()

