n=int(input("Enter number of enties: "))
Students = {}
for i in range(n):
    keys = input("Enter key: ")
    values = input("Enter value: ")
    Students[keys] = values
    print(Students)