import torch

def vertical_split(df):

    split = {

        "client1": ["HighBP","HighChol", "CholCheck"],
        "client2": ["Stroke","HeartDiseaseorAttack","PhysActivity", "AnyHealthcare", "DiffWalk", "GenHlth", "PhysHlth", "MentHlth"],
        "client3": ["Fruits","Veggies","HvyAlcoholConsump"], 
        "client4": ["Sex","Smoker", "Age", "Education", "Income", "BMI", "NoDocbcCost"]
    }

    tensors = {}

    for k,cols in split.items():

        tensors[k] = torch.tensor(
            df[cols].values,
            dtype=torch.float32
        )

    y = torch.tensor(
        df["Diabetes_binary"].values,
        dtype=torch.long
    )

    return tensors,y