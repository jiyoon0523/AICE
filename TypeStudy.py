# list
a = ['red', 'blue', 'green']
#a[0]= 'pink'#replaces 'red'
a.append('yellow')
a.insert(1, 'black')
b = ['purple', 'white']
a.extend(b)
print(a)

c = a+b
print(c)

d = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
d.remove(90)
print(d)

list1= ['a', 'bb', 'c', 'd', 'aaa', 'c', 'ddd', 'aaa', 'b', 'bb', 'd', 'aaa']
count= list1.count('aaa')
print(count)

list2= [1, -7, 5, 8, 3, 9, 11, 13]
list2.sort()
list2.sort(reverse= True)
print(list2)

d.pop(0)
print(d)

#tuple
t1= ('apple', 'banana', 'kiwi')
print(t1)

t2= 'banana', 'apple', 'kiwi'
print(t2)

# t2[0]= 'watermelon' #TypeError: 'tuple' object does not support item assignment

#dictionary
members= {'name': 'lindsay', 'email': 'jylindsay0523@gmail.com'}
print(members)

print(members.keys())
print(members.values())
print(members.items())
print(members.get('name'))
print('name' in members)
print('age' in members)
members.clear()
print(members)