n = 5
#Lower Triangular Pattern
print("Lower Triangular Pattern:")
for i in range(1, n+1):
    print('* ' * i)

#Upper Triangular Pattern
print("Upper Triangular Pattern:")
for i in range(n):
    print('  ' * i + '* ' * (n - i))

#Pyramid Pattern
print("Pyramid Triangular Pattern:")
for i in range(n):
    print('  ' * (n - i - 1) + '* ' * (2 * i + 1))

