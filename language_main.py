
#LIST
#Use a for loop to print the numbers from 1 to 20, inclusive.

numbers = list(range(1, 21))

for number in numbers:
    print(number)

#if-elif-else cahin that determines a person’s stage of life
age = 17

if age < 2:
    print("You're a baby!")
elif age < 4:
    print("You're a toddler!")
elif age < 13:
    print("You're a kid!")
elif age < 20:
    print("You're a teenager!")
elif age < 65:
    print("You're an adult!")
else:
    print("You're an elder!")


#Make a list of your favorite fruits, and then write a series of independent if statements that check for certain fruits in your list.
favorite_fruits = ['blueberries', 'salmonberries', 'peaches']

if 'bananas' in favorite_fruits:
    print("You really like bananas!")
if 'apples' in favorite_fruits:
    print("You really like apples!")
if 'blueberries' in favorite_fruits:
    print("You really like blueberries!")
if 'kiwis' in favorite_fruits:
    print("You really like kiwis!")
if 'peaches' in favorite_fruits:
    print("You really like peaches!")

#simulates how websites ensure that everyone has a unique username.
current_users = ['eric', 'willie', 'admin', 'erin', 'Ever']
new_users = ['sarah', 'Willie', 'PHIL', 'ever', 'Iona']

current_users_lower = [user.lower() for user in current_users]

for new_user in new_users:
    if new_user.lower() in current_users_lower:
        print("Sorry " + new_user + ", that name is taken.")
    else:
        print("Great, " + new_user + " is still available.")



#Write a program that asks the user what kind of rental car they would like. Print a message about that car, such as “Let me see if I can find you a Subaru”.
car1 = input("What kind of car would you like? ")
print("Let me see if I can find you a ", car1, ".")

#Write a program that asks the user how many people are in their dinner group. If the answer is more than eight, print a message saying they’ll have to wait for a table. Otherwise, report that their table is ready.
party_size = input("How many people are in your dinner party tonight? ")
party_size = int(party_size)

if party_size > 8:
    print("I'm sorry, you'll have to wait for a table.")
else:
    print("Your table is ready.")


'''Illustrate input and print.'''

applicant = input("Enter the applicant's name: ")
interviewer = input("Enter the interviewer's name: ")
time = input("Enter the appointment time: ")
print(interviewer, "will interview", applicant, "at", time)
