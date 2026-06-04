
import random, json
def run(n=100):
    n = int(n)
    first = ["Alex","Sam","Jordan","Taylor","Morgan","Casey","Riley","Jamie","Avery","Quinn",
             "Drew","Reese","Cameron","Skyler","Hayden","Rowan","Parker","Emerson","Finley","Sage",
             "Blake","Charlie","Dakota","Ellis","Frankie","Gray","Harper","Indigo","Justice","Kai"]
    last = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
            "Hernandez","Lopez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin","Lee",
            "Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson","Walker"]
    rows = [["Name","Score","Status"]]
    random.seed(42)
    for _ in range(n):
        name = f"{random.choice(first)} {random.choice(last)}"
        score = random.randint(0, 100)
        if score >= 60:
            status = random.choice(["Pass","Pass","Pass","Review"])
        elif score >= 40:
            status = random.choice(["Pending","Review","Fail"])
        else:
            status = "Fail"
        rows.append([name, score, status])
    return json.dumps(rows)
