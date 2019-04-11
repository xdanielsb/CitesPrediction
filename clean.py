import pandas as pd

nameFile = "data/filtered_data.xlsx"
data = pd.read_excel(nameFile, index_col=None)
numMissing = data.dropna()
rows, cols = numMissing.shape
headers = numMissing.columns.tolist()
dicti = open("data/dictionary","w+")

hashes = {}
for header in headers:
  hashes[ header ] = []
for header in headers:
  if( header == "Citedby"): continue

  uniques = numMissing[ header ].unique()
  key = 0
  for unique in uniques:
    hashes[ header].append( ( unique, key ))
    dicti.write("{},{}\n".format(unique, key))
    try:
        numMissing.replace(to_replace= unique, value=key, inplace = True)
        key += 1
    except Exception as e:
        print("")
dicti.close()
print( headers )
print( data )
numMissing["Citedby"] = numMissing["Citedby"].astype(int)

numMissing.to_excel("data/cleaned.xlsx", engine="xlsxwriter")
