keywords = []
with open('/Users/wanzheng/Desktop/local-embedding/data/dblp/non-para/repre_cand_0.txt') as file:
    for i in file:
        keywords.append(i[:-1])
print keywords

specific_terms = []

for j in range(5):
    with open('/Users/wanzheng/Desktop/local-embedding/data/dblp/non-para/object_recognition/repre_cand_' + str(j) + '.txt') as file:
        num_line = sum(1 for line in file)
    with open('/Users/wanzheng/Desktop/local-embedding/data/dblp/non-para/object_recognition/repre_cand_' + str(j) + '.txt') as file:
        num = 0
        for i in file:
            if num < num_line*0.5:
                specific_terms.append(i[:-1])
                num += 1

print specific_terms

for i in keywords:
    if i not in specific_terms:
        print i
