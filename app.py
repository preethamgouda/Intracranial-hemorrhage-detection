import os
import json


def signup():
    first_name = input("Enter your first name: ")
    middle_name = input("Enter your middle name: ")
    last_name = input("Enter your last name: ")
    dob = input("Enter your date of birth (YYYY-MM-DD): ")
    age = input("Enter your age: ")
    phone_number = input("Enter your phone number: ")
    aadhar_number = input("Enter your Aadhaar number: ")
    pan_number = input("Enter your PAN number: ")
    password = input("Enter your password: ")

    data = {
        "first_name": first_name,
        "middle_name": middle_name,
        "last_name": last_name,
        "dob": dob,
        "age": age,
        "phone_number": phone_number,
        "aadhar_number": aadhar_number,
        "pan_number": pan_number,
        "password": password,
    }

    with open("users.txt", "a") as file:
        file.write(json.dumps(data) + "\n")


def login():
    """Logs in a user and returns user data if successful."""
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    with open("users.txt", "r") as file:
        for line in file:
            data = json.loads(line)
            if data["first_name"] == username and data["password"] == password:
                return data

    return None


def account_exists(username):
    """iduuu check madoke weather the user name exists"""
    with open("users.txt", "r") as file:
        for line in file:
            data = json.loads(line)
            if data["first_name"] == username:
                return True

    return False


def create_users_file():
    """e code check madate weather file user.txt exist ide no ilvo anta in the directory if exists it will use it otherwise it will create new one"""
    if not os.path.exists("users.txt"):
        with open("users.txt", "w") as file:
            pass


def display_user_details(user_data):
    """idu optional code if you want to display details after login other wise comment out this"""
    print("User Details:")
    print("-------------")
    print("First Name:", user_data["first_name"])
    print("Middle Name:", user_data["middle_name"])
    print("Last Name:", user_data["last_name"])
    print("Date of Birth:", user_data["dob"])
    print("Age:", user_data["age"])
    print("Phone Number:", user_data["phone_number"])
    print("Aadhaar Number:", user_data["aadhar_number"])
    print("PAN Number:", user_data["pan_number"])


def main():
    """The main function."""
    create_users_file()

    while True:
        print("1. Sign up")
        print("2. Login")
        option = input("Enter your option: ")

        if option == "1":
            username = input("Enter your username: ")
            if account_exists(username):
                print("Account already exists!")
            else:
                signup()
                print("Account created successfully!")

        elif option == "2":
            user_data = login()
            if user_data:
                print("Login successful!")
                display_user_details(user_data)
            else:
                print("Login failed!")

        else:
            break


if __name__ == "__main__":
    main()
