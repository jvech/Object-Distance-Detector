import wget, os

if __name__ == "__main__":
    ls = os.listdir('weights')
    files = [file for file in ls if file[-8:-1] in ".weights"]
    print(files)
    with open("weights/links.txt", "r") as f:
        links = [line.strip() for line in f.readlines()]
    
    for x in files:
        for y in links:
            if x in y:
                links.remove(y)
    if links != []:
        for i in links:
            wget.download(i, 'weights/')
    else:
        print("Nothing to download")