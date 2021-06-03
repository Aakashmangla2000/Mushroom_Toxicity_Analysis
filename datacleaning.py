import pandas as pd 
import numpy as np

data = pd.DataFrame()

def clean_main(data):
    
    # values={"bell":0,"conical":1,"convex":"convex","flat":2,"knobbed":3,"sunken":4}
    # data["cap-shape"]=data["cap-shape"].replace(values)
    values2={"fibrous":0,"grooves":1,"smooth":2,"scaly":3}
    data["cap-surface"]=data["cap-surface"].replace(values2)
    # values3={"brown":4,"buff":0,"cinnamon":1,"gray":3,"green":6,"pink":5,"purple":7,"red":2,"white":8,"yellow":9}
    # data["cap-color"]=data["cap-color"].replace(values3)
    values4={"almond":0,"anise":3,"creosote":1,"fishy":8,"foul":2, "musty":4,"none":5,"pungent":6,"spicy":7}
    data["odor"]=data["odor"].replace(values4)
    #values5={"attached":0,"descending":1,"free":2,"notched":3}
    # data["gill-attachment"]=data["gill-attachment"].replace(values5)
    values6={"close":0,"crowded":1,"distant":2}
    data["gill-spacing"]=data["gill-spacing"].replace(values6)
    values7={"broad":0,"narrow":1}
    data["gill-size"]=data["gill-size"].replace(values7)
    values8={"black":4,"brown":5,"buff":0,"chocolate":3,"gray":2,"green":8,"orange":6,"pink":7,"purple":9,"red":1, "white":10,"yellow":11}
    data["gill-color"]=data["gill-color"].replace(values8)
    # values9={"enlarging":0,"tapering":1}
    # data["stalk-shape"]=data["stalk-shape"].replace(values9)
    values10={"bulbous":1,"club":2,"cup":5,"equal":3,"rhizomorphs":6,"rooted":4,"missing":0}
    data["stalk-root"]=data["stalk-root"].replace(values10)
    values11={"fibrous":0,"scaly":3,"silky":1,"smooth":2}
    data["stalk-surface-above-ring"]=data["stalk-surface-above-ring"].replace(values11)
    data["stalk-surface-below-ring"]=data["stalk-surface-below-ring"].replace(values11)
    values12={"buff":0,"cinnamon":1,"red":2,"gray":3, "brown":4,"orange":5, "pink":6, "white":7,"yellow":8}
    data["stalk-color-above-ring"]=data["stalk-color-above-ring"].replace(values12)
    data["stalk-color-below-ring"]=data["stalk-color-below-ring"].replace(values12)
    # veil_type={"partial":0,"universal":1} 
    # data["veil-type"]=data["veil-type"].replace(veil_type)
    # veil_color={"brown":0,"orange":1,"white":2,"yellow":3} 
    # data["veil-color"]=data["veil-color"].replace(veil_color)
    # ring_number= {"none":0,"one":1,"two":2}
    # data["ring-number"]=data["ring-number"].replace(ring_number)
    ring_type={"cobwebby":0,"evanescent":1,"flaring":2,"large":3,"none":4,"pendant":5,"sheathing":6,"zone":7}
    data["ring-type"]=data["ring-type"].replace(ring_type)
    spore_print_color= {"black":2,"brown":3,"buff":0,"chocolate":1,"green":5,"orange":4,"purple":6,"white":7,"yellow":8}
    data["spore-print-color"]=data["spore-print-color"].replace(spore_print_color)
    population={"abundant":0,"clustered":1,"numerous":2,"scattered":3,"several":4,"solitary":5}
    data["population"]=data["population"].replace(population)
    habitat={"grasses":0,"leaves":1,"meadows":2,"paths":3,"urban":4,"waste":5,"woods":6}
    data["habitat"]=data["habitat"].replace(habitat)
    bruises={"true":1,"false":0}
    data["bruises"]=data["bruises"].replace(bruises)
    
    return data










