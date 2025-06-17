import csv


with open('predictive_maintenance.csv', 'r') as file:
    reader = csv.reader(file, delimiter = '\t')
    for row in reader:
        a=row[0].split(",")
        str=""
        if(a[2]=="L"):
            str+="1,"
        elif(a[2]=="M"):
            str+="2,"
        else:
            str+="3,"
        
        str+=a[3]+","+a[4]+","+a[5]+","+a[6]+","+a[7]+","+a[8]
        
        if(a[9]=="No Failure"):
            str+=",0"
        else:
            str+=",1"
        with open("data.txt", "a") as myfile:
            myfile.write(str+"\n")