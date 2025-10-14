import json

# Data extracted from the previous response (20 entities, 10 queries each)
# Note: Attributes are cleaned up slightly to remove parentheses/extra words for consistency.
data = [
    {"entity": "Car", "attributes": "model, year, color", "queries": [
        "Get the car's model, year, and color.",
        "List the model, year, and color of the car.",
        "Identify what model, year, and color the car is.",
        "Retrieve the car's model designation, production year, and exterior color.",
        "Can you tell me the model, year, and color specifications for this vehicle?",
        "What are the model, year, and color of the car?",
        "Please provide the car's model, year, and color.",
        "Extract the model name, manufacturing year, and paint color of the car.",
        "Specify the car's model, year, and color.",
        "Document the car's model, year, and color details."
    ]},
    {"entity": "Recipe", "attributes": "ingredients, preparation time, serving size", "queries": [
        "Extract all ingredients, the total preparation time, and the serving size for the recipe.",
        "List the recipe's ingredients, prep time, and serving yield.",
        "Retrieve the ingredients, how long it takes to prepare, and the portion amount.",
        "What are the ingredients, preparation time, and serving size of the dish?",
        "Please provide the full list of ingredients, the prep time, and how many people the recipe serves.",
        "Gather the ingredients list, the estimated prep duration, and the serving size from the recipe.",
        "Identify the required ingredients, the preparation time, and the recipe's serving size.",
        "Document the recipe's ingredients, time needed for preparation, and the number of servings.",
        "Determine the ingredients, preparation time, and serving size for the recipe.",
        "Find the ingredients, preparation time, and serving size of this culinary guide."
    ]},
    {"entity": "Employee", "attributes": "full name, department, salary", "queries": [
        "Get the employee's full name, department, and salary.",
        "Retrieve the full name, assigned department, and annual salary of the employee.",
        "List the employee's full name, department, and pay.",
        "What is the employee's full name, which department do they work in, and what is their salary?",
        "Please provide the employee's complete name, their department affiliation, and their compensation amount.",
        "Extract the full name, department, and salary information for the employee.",
        "Can you find the employee's full name, the department they belong to, and their current salary?",
        "Identify the full name, department, and salary associated with this employee.",
        "Specify the employee's full name, work department, and rate of pay.",
        "Pull out the employee's full name, department, and salary details."
    ]},
    {"entity": "Building", "attributes": "address, number of floors, primary use", "queries": [
        "Identify the building's address, number of floors, and primary use.",
        "List the address, floor count, and main function of the structure.",
        "Retrieve the building's street address, total number of levels, and its chief purpose.",
        "What are the building's address, the quantity of floors, and its primary utility?",
        "Please provide the address, the number of floors, and the primary use of the building.",
        "Extract the address, number of floors, and main operational use of the building.",
        "Document the building's location (address), floor number, and main usage type.",
        "Gather the address, number of stories, and primary function of the building.",
        "Determine the building's address, floor count, and intended use.",
        "Fetch the address, number of floors, and primary use of this structure."
    ]},
    {"entity": "Smartphone", "attributes": "key features, battery life, screen size", "queries": [
        "What are the key features, battery life, and screen size of the smartphone?",
        "List the main features, battery longevity, and display size of the smartphone.",
        "Identify the important features, how long the battery lasts, and the smartphone's screen size.",
        "Retrieve the smartphone's key specifications, expected battery life, and screen diagonal measurement.",
        "Please provide the smartphone's core features, battery life duration, and screen size.",
        "Extract the key features, battery life, and screen size details from the smartphone specifications.",
        "Show me the smartphone's standout features, battery life, and size of the screen.",
        "Document the device's key features, battery capacity/life, and screen dimensions.",
        "Get the smartphone's main features, battery run time, and screen size.",
        "Specify the key features, battery life estimate, and screen size of the phone."
    ]},
    {"entity": "Patient", "attributes": "patient ID, date of birth, assigned physician", "queries": [
        "Please provide the patient's ID, date of birth, and assigned physician.",
        "Retrieve the patient ID, DOB, and the name of the assigned doctor.",
        "List the patient's identifier, birth date, and physician in charge.",
        "Get the patient's ID number, their date of birth, and who their assigned physician is.",
        "Extract the patient identification number, date of birth, and assigned physician's name.",
        "Identify the patient ID, the date they were born, and the name of the physician assigned to them.",
        "Document the patient's ID, date of birth, and the details of the assigned doctor.",
        "What are the patient's ID, date of birth, and who is the assigned doctor?",
        "Fetch the patient's ID, their date of birth, and the assigned physician.",
        "Record the patient's ID, date of birth, and the attending physician."
    ]},
    {"entity": "Movie", "attributes": "title, release year, director, run time", "queries": [
        "Gather the movie's title, release year, director, and run time.",
        "List the film's title, year it was released, director's name, and its duration.",
        "Retrieve the movie's title, release date year, director, and the length of the running time.",
        "Identify the title, year of release, director, and run time of the film.",
        "Please provide the movie's title, the year it came out, the director, and its total run time.",
        "Extract the title, release year, director, and run time from the movie's details.",
        "What are the movie's title, release year, director, and total run time?",
        "Document the film's title, release year, director, and duration.",
        "Get the movie title, year of release, director, and run time.",
        "Find the title, release year, director, and run time for the movie."
    ]},
    {"entity": "City", "attributes": "population, area in square miles, major industries", "queries": [
        "Can you find the city's population, area in square miles, and major industries?",
        "List the population, area in square miles, and primary industries of the city.",
        "Retrieve the city's current population, its size (area in sq. miles), and key industries.",
        "What are the city's population count, its area in square miles, and its most important industries?",
        "Please provide the city's population, area in sq. miles, and major industrial sectors.",
        "Extract the population, the area (in square miles), and the major industries from the city data.",
        "Identify the city's population, its geographical area (sq. miles), and its major industries.",
        "Document the city's population figures, the area it covers in square miles, and its major industries.",
        "Gather the population, area in square miles, and major industries of the city.",
        "Get the city's population, the area in square miles, and its leading industries."
    ]},
    {"entity": "Song", "attributes": "artist, album title, genre", "queries": [
        "Pull out the song's artist, album title, and genre.",
        "List the song's performing artist, the album it's on, and its genre type.",
        "Retrieve the name of the artist, the album title, and the genre of the track.",
        "Identify the song's artist, the associated album title, and its musical genre.",
        "Please provide the song's artist, the album title, and its genre.",
        "Extract the artist, album title, and genre from the song's metadata.",
        "What are the song's artist, album title, and music genre?",
        "Get the song's artist, the title of the album, and its genre.",
        "Specify the song's artist, album title, and genre.",
        "Document the song's artist, album title, and genre information."
    ]},
    {"entity": "Chemical Compound", "attributes": "formula, molecular weight, state of matter", "queries": [
        "Determine the chemical compound's formula, molecular weight, and state of matter.",
        "List the compound's chemical formula, its molecular weight, and its physical state.",
        "Retrieve the chemical formula, molecular weight, and state of matter for the compound.",
        "Identify the formula, molecular weight, and state of matter of the chemical compound.",
        "Please provide the compound's formula, the weight of its molecule, and its state of matter.",
        "Extract the chemical formula, molecular weight, and state of matter for the compound.",
        "What are the compound's formula, molecular weight, and state of matter (solid, liquid, gas)?",
        "Get the chemical compound's formula, molecular weight, and state of matter.",
        "Document the compound's formula, molecular weight, and state of matter details.",
        "Specify the chemical formula, molecular weight, and state of matter for the compound."
    ]},
    {"entity": "Restaurant", "attributes": "rating, average price range, cuisine type", "queries": [
        "Show me the restaurant's rating, average price range, and cuisine type.",
        "List the restaurant's rating, price range, and the type of food (cuisine).",
        "Retrieve the rating, the average cost bracket, and the cuisine type served at the restaurant.",
        "Identify the restaurant's rating, average price range, and its type of cuisine.",
        "Please provide the restaurant's customer rating, the typical price range, and its cuisine type.",
        "Extract the rating, average price range, and cuisine type from the restaurant information.",
        "What are the restaurant's rating, average price, and the kind of cuisine it offers?",
        "Get the restaurant's rating, price range, and cuisine type.",
        "Determine the restaurant's rating, average price range, and cuisine.",
        "Document the restaurant's rating, its average price range, and the cuisine type served."
    ]},
    {"entity": "Laptop", "attributes": "warranty period, product number, manufacturer", "queries": [
        "Specify the warranty period, product number, and manufacturer of the laptop.",
        "List the laptop's warranty duration, its product number, and the company that made it.",
        "Retrieve the warranty period, the specific product number, and the manufacturer's name.",
        "Identify the laptop's warranty period, its unique product number, and the manufacturer.",
        "Please provide the laptop's warranty period, the product number, and who manufactured it.",
        "Extract the warranty period, product number, and manufacturer from the laptop's details.",
        "What are the laptop's warranty period, product number, and manufacturer?",
        "Get the laptop's warranty length, product number, and manufacturer.",
        "Document the laptop's warranty period, product number, and manufacturer.",
        "Find the warranty period, the product number, and the manufacturer of the laptop."
    ]},
    {"entity": "Stock", "attributes": "ticker symbol, current price, daily trading volume", "queries": [
        "Obtain the stock's ticker symbol, current price, and daily trading volume.",
        "List the stock's ticker, its current market price, and the daily volume of trade.",
        "Retrieve the stock ticker, the present price, and the trading volume for the day.",
        "Identify the stock's ticker symbol, its real-time price, and the daily trading volume.",
        "Please provide the stock's ticker symbol, the current price, and the daily trading volume data.",
        "Extract the ticker symbol, current price, and daily trading volume from the stock quote.",
        "What are the stock's ticker symbol, its current price, and its daily trading volume?",
        "Get the stock's ticker, the current price, and the daily volume.",
        "Document the stock's ticker symbol, current price, and daily trading volume.",
        "Specify the stock's ticker symbol, current price, and the volume of daily trade."
    ]},
    {"entity": "Planet", "attributes": "diameter, orbital period, number of moons", "queries": [
        "Document the planet's diameter, orbital period, and number of moons.",
        "List the planet's diameter, the time it takes to orbit (orbital period), and the count of its moons.",
        "Retrieve the planet's diameter, its orbital period, and how many moons it has.",
        "Identify the planet's diameter, the length of its orbit, and the number of satellites (moons).",
        "Please provide the planet's diameter measurement, its orbital period, and the quantity of moons.",
        "Extract the planet's diameter, orbital period, and number of moons.",
        "What are the planet's diameter, orbital period, and number of moons?",
        "Get the planet's diameter, orbital period, and number of moons.",
        "Determine the planet's diameter, orbital period, and moon count.",
        "Specify the planet's diameter, orbital period, and number of moons."
    ]},
    {"entity": "Artwork", "attributes": "title, artist, year of creation", "queries": [
        "Fetch the artwork's title, artist, and year of creation.",
        "List the title, the artist's name, and the creation year of the artwork.",
        "Retrieve the artwork's title, the artist who created it, and the year it was made.",
        "Identify the title, artist, and creation year for the piece of art.",
        "Please provide the artwork's title, the artist, and the year it was completed.",
        "Extract the title, artist, and year of creation from the artwork's details.",
        "What are the artwork's title, artist, and the year it was created?",
        "Get the artwork's title, artist, and year of creation.",
        "Document the artwork's title, artist, and the year it was finished.",
        "Specify the title, artist, and year of creation of the artwork."
    ]},
    {"entity": "Course", "attributes": "course name, course code, credit hours", "queries": [
        "Could you tell me the course's name, course code, and credit hours?",
        "List the name, code, and number of credit hours for the course.",
        "Retrieve the course name, the corresponding course code, and the total credit hours.",
        "Identify the course's full name, its official code, and the assigned credit hours.",
        "Please provide the name of the course, the course code, and the credit hours.",
        "Extract the course name, course code, and credit hours from the course information.",
        "What are the course's name, code, and credit hours?",
        "Get the course name, course code, and the number of credit hours.",
        "Document the course's name, course code, and credit hour count.",
        "Specify the course name, course code, and credit hours."
    ]},
    {"entity": "Software", "attributes": "version number, license type, installation date", "queries": [
        "Record the software's version number, license type, and installation date.",
        "List the software's version number, the type of license, and when it was installed.",
        "Retrieve the version number, license type, and the date of installation for the software.",
        "Identify the software's version number, its license category, and the date it was set up.",
        "Please provide the software's version number, the license type, and the installation date.",
        "Extract the version number, license type, and installation date of the software.",
        "What are the software's version number, license type, and installation date?",
        "Get the software's version number, license type, and installation date.",
        "Document the software's version number, license type, and installation date.",
        "Specify the version number, license type, and installation date for the software."
    ]},
    {"entity": "Animal", "attributes": "species, average lifespan, native habitat", "queries": [
        "Locate the animal's species, average lifespan, and native habitat.",
        "List the animal's species, its typical lifespan, and where it naturally lives (native habitat).",
        "Retrieve the animal species, the average number of years it lives, and its native habitat.",
        "Identify the animal's species, its average lifespan, and its original habitat.",
        "Please provide the animal's species, average lifespan, and native habitat.",
        "Extract the species, average lifespan, and native habitat from the animal's profile.",
        "What are the animal's species, its average lifespan, and its native habitat?",
        "Get the animal's species, average lifespan, and native habitat.",
        "Determine the animal's species, average lifespan, and natural habitat.",
        "Document the animal's species, average lifespan, and native habitat details."
    ]},
    {"entity": "Project", "attributes": "start date, expected completion date, current status", "queries": [
        "Report on the project's start date, expected completion date, and current status.",
        "List the project's start date, its anticipated completion date, and the present status.",
        "Retrieve the start date, the expected date of completion, and the current status of the project.",
        "Identify the project's start date, the target completion date, and its current status.",
        "Please provide the project's start date, expected completion date, and current status.",
        "Extract the project's start date, expected completion date, and current status.",
        "What are the project's start date, expected completion date, and current status?",
        "Get the project's start date, expected completion date, and current status.",
        "Document the project's start date, the expected end date, and its current status.",
        "Find the project's start date, expected completion date, and current status."
    ]},
    {"entity": "Website", "attributes": "url, domain age, primary language", "queries": [
        "Give me the website's URL, domain age, and primary language.",
        "List the website's URL, the age of its domain, and its main language.",
        "Retrieve the URL, how old the domain is (domain age), and the primary language of the website.",
        "Identify the website's URL, the domain's age, and the main language used.",
        "Please provide the website's URL, domain age, and its primary language.",
        "Extract the website's URL, domain age, and primary language details.",
        "What are the website's URL, domain age, and primary language?",
        "Get the website's URL, domain age, and primary language.",
        "Document the website's URL, domain age, and primary language.",
        "Specify the URL, domain age, and primary language of the website."
    ]}
]

# Fixed instruction string
fixed_instruction = "Extract the main entity (noun) and attributes/properties from the following query."

# Final list to hold all structured JSON records
training_data = []

# Process the data
for item in data:
    # Convert entity name to lowercase and replace spaces with underscores for consistency
    entity = item["entity"].lower().replace(' ', '_')
    attributes = item["attributes"]
    
    # Custom format string for the "output" field, using escaped quotes as requested
    output_format = f'\"entity\":\"{entity}\", \"attributes\": \"{attributes}\"'
    
    for query in item["queries"]:
        record = {
            "instruction": fixed_instruction,
            "input": query,
            "output": output_format
        }
        training_data.append(record)

# Save the result to training_query.json
output_filename = "training_query.json"
with open(output_filename, 'w', encoding='utf-8') as f:
    # Use json.dumps with indent=2 for readability, though it doesn't strictly matter for the model's output
    # The output string is specially formatted to match the user's request:
    # "output": "\"entity\":\"website\", \"attributes\": \"URL, domain age, primary language\""
    # To achieve this, the output value is NOT parsed as JSON, but treated as a literal string value,
    # and the quotes inside it are escaped (\\").
    json.dump(training_data, f, ensure_ascii=False, indent=2)

print(f"Successfully created {output_filename} with {len(training_data)} records.")
print("\nFirst record example (note the escaped quotes in the 'output' field):")
print(json.dumps(training_data[0], indent=2))