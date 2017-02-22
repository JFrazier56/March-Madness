import numpy as np
import pandas as pd

def processData():
    detailed_results = pd.read_csv('RegularSeasonDetailedResults.csv')
    df_1 = pd.DataFrame()
    df_1[['team1', 'team2']] = detailed_results[['Wteam', 'Lteam']].copy()
    df_1['pred'] = 1
    df_2 = pd.DataFrame()
    df_2[['team1', 'team2']] = detailed_results[['Lteam', 'Wteam']].copy()
    df_2['pred'] = 0
    df = pd.concat((df_1, df_2), axis=0)

    train = df.values
    np.random.shuffle(train)


    print(df)



def main():
    processData()


if __name__ == "__main__":
    main()
