import os
import streamlit as st
from typing import Dict, List, Optional
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Initialize Streamlit app settings
st.set_page_config(page_title="Travel Itinerary Planner", page_icon="✈️", layout="wide")
# Get API key
with open("groq_api_key.txt", "r") as file:
    api_key = file.read().strip()
# Initialize LLM with Groq
os.environ["GROQ_API_KEY"] = api_key # Use your API key from Streamlit secrets
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama3-70b-8192",  # Use appropriate Groq model
        temperature=0.7,
        max_tokens=4096,
    )

# Helper function to check if we have sufficient info to generate an itinerary
def has_sufficient_info(travel_info):
    return (travel_info.get("destination") and 
            (travel_info.get("duration") or 
            (travel_info.get("start_date") and travel_info.get("end_date"))))

# Create the travel information extraction chain
def create_extract_info_chain(llm):
    extract_info_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a travel planning assistant. Your task is to extract travel information from the user's message.
        Extract relevant travel information including:
        - Destination and origin locations
        - Dates or duration of travel
        - Budget information
        - Purpose of travel
        - Traveler preferences
        - Number of travelers
        - Any special requirements
        
        Format your response as a simple structured text with clear labels for each piece of information.
        If information is not provided, indicate it as "Not specified".
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="{input}")
    ])

    return LLMChain(
        llm=llm,
        prompt=extract_info_prompt,
        verbose=True
    )

# Create follow-up questions chain
def create_follow_up_chain(llm):
    follow_up_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a travel planning assistant. Based on the information already collected, 
        generate 2-3 relevant follow-up questions to fill in missing details about the user's travel plans.
        
        Current information:
        {current_info}
        
        Focus on missing information that would be important for creating a detailed travel itinerary, such as:
        - Missing destination or origin information
        - Missing dates or duration
        - Unclear budget expectations
        - Specific interests or preferences (food, activities, accommodation)
        - Mobility considerations or special requirements
        - Number of travelers and their relationships
        
        Keep your response concise and focused only on the questions.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="Please help me fill in missing travel details.")
    ])

    return LLMChain(
        llm=llm,
        prompt=follow_up_prompt,
        verbose=True
    )

# Create itinerary generation chain
def create_itinerary_chain(llm):
    generate_itinerary_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert travel planner. Create a detailed day-by-day itinerary based on the user's travel information.
        
        Travel Information:
        {travel_info}
        
        Create a logical, well-paced itinerary that:
        1. Groups activities by proximity to minimize travel time
        2. Includes appropriate meal breaks at quality establishments matching the user's preferences
        3. Balances tourist attractions with authentic local experiences
        4. Respects the user's budget constraints
        5. Accommodates any mobility constraints or special requirements
        6. Provides specific timing for activities to create a realistic schedule
        7. Includes helpful local tips specific to the destination
        
        Format your response as a markdown document with:
        - A title and brief summary
        - Day-by-day breakdown with morning, afternoon, and evening activities
        - Meal recommendations for each day
        - Accommodation details
        - Local tips
        - Estimated total cost
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessage(content="Please generate a detailed itinerary for my trip.")
    ])

    return LLMChain(
        llm=llm,
        prompt=generate_itinerary_prompt,
        verbose=True
    )

# Helper function to parse travel info from extraction output
def parse_travel_info(extracted_text):
    travel_info = {
        "destination": None,
        "origin": None,
        "start_date": None,
        "end_date": None,
        "duration": None,
        "budget": None,
        "purpose": None,
        "traveler_count": None,
        "preferences": {
            "cuisine": None,
            "activities": None,
            "accommodation": None,
            "mobility": None,
            "special_requirements": None
        }
    }
    
    lines = extracted_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if ":" not in line:
            continue
            
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        
        if value == "Not specified" or not value:
            continue
            
        if "destination" in key:
            travel_info["destination"] = value
        elif "origin" in key or "starting" in key or "from" in key:
            travel_info["origin"] = value
        elif "start date" in key:
            travel_info["start_date"] = value
        elif "end date" in key:
            travel_info["end_date"] = value
        elif "duration" in key or "length" in key:
            # Try to extract a number
            import re
            numbers = re.findall(r'\d+', value)
            if numbers:
                travel_info["duration"] = int(numbers[0])
            else:
                travel_info["duration"] = value
        elif "budget" in key:
            travel_info["budget"] = value
        elif "purpose" in key:
            travel_info["purpose"] = value
        elif "travelers" in key or "people" in key:
            # Try to extract a number
            import re
            numbers = re.findall(r'\d+', value)
            if numbers:
                travel_info["traveler_count"] = int(numbers[0])
            else:
                travel_info["traveler_count"] = value
        elif "food" in key or "cuisine" in key:
            travel_info["preferences"]["cuisine"] = value
        elif "activ" in key or "interest" in key:
            travel_info["preferences"]["activities"] = value
        elif "accommod" in key or "hotel" in key or "stay" in key:
            travel_info["preferences"]["accommodation"] = value
        elif "mobility" in key or "access" in key:
            travel_info["preferences"]["mobility"] = value
        elif "special" in key or "requirement" in key:
            travel_info["preferences"]["special_requirements"] = value
            
    return travel_info

# Helper function to merge new information into existing travel info
def merge_travel_info(existing_info, new_info):
    if not existing_info:
        return new_info
        
    result = existing_info.copy()
    
    for key, value in new_info.items():
        if key == "preferences" and value and existing_info.get("preferences"):
            # Merge preferences
            for pref_key, pref_value in value.items():
                if pref_value:
                    result["preferences"][pref_key] = pref_value
        elif value:  # Only update if the new value is not None/empty
            result[key] = value
            
    return result

# Helper function to format travel info summary for display
def format_travel_info_summary(travel_info):
    summary = []
    
    if travel_info.get("destination"):
        summary.append(f"📍 **Destination**: {travel_info['destination']}")
    
    if travel_info.get("origin"):
        summary.append(f"🏠 **Starting from**: {travel_info['origin']}")
    
    if travel_info.get("start_date") and travel_info.get("end_date"):
        summary.append(f"📅 **Dates**: {travel_info['start_date']} to {travel_info['end_date']}")
    elif travel_info.get("duration"):
        summary.append(f"⏱️ **Duration**: {travel_info['duration']} days")
    
    if travel_info.get("budget"):
        summary.append(f"💰 **Budget**: {travel_info['budget']}")
    
    if travel_info.get("purpose"):
        summary.append(f"🎯 **Purpose**: {travel_info['purpose']}")
    
    if travel_info.get("traveler_count"):
        summary.append(f"👥 **Number of travelers**: {travel_info['traveler_count']}")
    
    if travel_info.get("preferences"):
        prefs = travel_info["preferences"]
        if prefs.get("cuisine"):
            summary.append(f"🍽️ **Food preferences**: {prefs['cuisine']}")
        
        if prefs.get("activities"):
            summary.append(f"🎭 **Activity interests**: {prefs['activities']}")
        
        if prefs.get("accommodation"):
            summary.append(f"🏨 **Accommodation preferences**: {prefs['accommodation']}")
        
        if prefs.get("mobility"):
            summary.append(f"🚶 **Mobility considerations**: {prefs['mobility']}")
        
        if prefs.get("special_requirements"):
            summary.append(f"✨ **Special requirements**: {prefs['special_requirements']}")
    
    return "\n".join(summary)

# Get missing details for follow-up questions
def get_missing_details(travel_info):
    missing = []
    
    if not travel_info.get("destination"):
        missing.append("destination")
    
    if not travel_info.get("duration") and not (travel_info.get("start_date") and travel_info.get("end_date")):
        missing.append("travel dates or duration")
    
    if not travel_info.get("budget"):
        missing.append("budget")
        
    if not travel_info.get("traveler_count"):
        missing.append("number of travelers")
    
    if not travel_info.get("purpose"):
        missing.append("purpose of your trip")
    
    preferences = travel_info.get("preferences", {})
    if not preferences.get("activities"):
        missing.append("activity preferences")
    
    if not preferences.get("accommodation"):
        missing.append("accommodation preferences")
        
    if not preferences.get("cuisine"):
        missing.append("food preferences")
    
    return missing

# Initialize Streamlit app
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "travel_info" not in st.session_state:
        st.session_state.travel_info = {
            "destination": None,
            "origin": None,
            "start_date": None,
            "end_date": None,
            "duration": None,
            "budget": None,
            "purpose": None,
            "traveler_count": None,
            "preferences": {
                "cuisine": None,
                "activities": None,
                "accommodation": None,
                "mobility": None,
                "special_requirements": None
            }
        }
    
    if "itinerary" not in st.session_state:
        st.session_state.itinerary = None
    
    if "stage" not in st.session_state:
        st.session_state.stage = "greeting"
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    # Initialize session state
    init_session_state()
    
    # Page title and description
    st.title("✈️ Travel Itinerary Planner")
    st.markdown("""
    Welcome to your AI travel assistant! Share your travel plans and preferences, 
    and I'll create a personalized day-by-day itinerary for you.
    """)
    
    # Initialize LLM and chains
    llm = get_llm()
    extract_info_chain = create_extract_info_chain(llm)
    follow_up_chain = create_follow_up_chain(llm)
    itinerary_chain = create_itinerary_chain(llm)
    
    # Sidebar for showing current travel information
    with st.sidebar:
        st.subheader("Your Travel Details")
        if st.session_state.travel_info:
            st.markdown(format_travel_info_summary(st.session_state.travel_info))
        
        # Add reset button
        if st.button("Start New Trip"):
            st.session_state.messages = []
            st.session_state.travel_info = {
                "destination": None,
                "origin": None,
                "start_date": None,
                "end_date": None,
                "duration": None,
                "budget": None,
                "purpose": None,
                "traveler_count": None,
                "preferences": {
                    "cuisine": None,
                    "activities": None,
                    "accommodation": None,
                    "mobility": None,
                    "special_requirements": None
                }
            }
            st.session_state.itinerary = None
            st.session_state.stage = "greeting"
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initial greeting
    if st.session_state.stage == "greeting" and not st.session_state.messages:
        initial_greeting = """
        Hi there! 👋 I'm your personal travel planner. I'll help you create a customized itinerary based on your preferences.

        To get started, could you please share some basic details about your trip:
        - Where are you traveling to?
        - When are you planning to travel and for how long?
        - What's your approximate budget for this trip?
        - Is this trip for leisure, business, family vacation, or something else?
        - Do you have any specific interests or preferences (e.g., history, food, nature)?

        The more details you provide, the better I can tailor your itinerary!
        """
        
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
        st.session_state.chat_history.append(AIMessage(content=initial_greeting))
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("What are your travel plans?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(HumanMessage(content=prompt))
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process user message
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Process based on current stage
                if st.session_state.stage == "greeting":
                    # After greeting, move to collecting info
                    st.session_state.stage = "collecting_info"
                    
                    # Extract information from user message
                    extracted_info_text = extract_info_chain.run(
                        input=prompt,
                        chat_history=st.session_state.chat_history
                    )
                    
                    # Parse the extracted information
                    extracted_info = parse_travel_info(extracted_info_text)
                    
                    # Update travel info with extracted information
                    st.session_state.travel_info = merge_travel_info(st.session_state.travel_info, extracted_info)
                    
                    # Get specific missing details
                    missing_details = get_missing_details(st.session_state.travel_info)
                    
                    # Generate follow-up questions for missing information
                    follow_ups = follow_up_chain.run(
                        current_info=format_travel_info_summary(st.session_state.travel_info),
                        chat_history=st.session_state.chat_history
                    )
                    
                    response = f"Thanks for sharing those details! I've noted:\n\n"
                    response += format_travel_info_summary(st.session_state.travel_info)
                    
                    if missing_details:
                        response += "\n\n**I still need the following information to create your itinerary:**\n"
                        for detail in missing_details:
                            response += f"- {detail.capitalize()}\n"
                    
                    response += "\n\n" + follow_ups
                    
                    # Check if we have enough info to generate an itinerary
                    if has_sufficient_info(st.session_state.travel_info) and len(missing_details) <= 2:
                        response += "\n\nWould you like me to generate your itinerary now, or would you prefer to add more details?"
                        st.session_state.stage = "generating_itinerary"
                
                elif st.session_state.stage == "collecting_info":
                    # Continue collecting information
                    extracted_info_text = extract_info_chain.run(
                        input=prompt,
                        chat_history=st.session_state.chat_history
                    )
                    
                    # Parse the extracted information
                    extracted_info = parse_travel_info(extracted_info_text)
                    
                    # Update travel info with extracted information
                    st.session_state.travel_info = merge_travel_info(st.session_state.travel_info, extracted_info)
                    
                    # Get specific missing details
                    missing_details = get_missing_details(st.session_state.travel_info)
                    
                    response = f"I've updated your travel details:\n\n"
                    response += format_travel_info_summary(st.session_state.travel_info)
                    
                    # Check if we have enough info to generate an itinerary
                    if has_sufficient_info(st.session_state.travel_info) and len(missing_details) <= 2:
                        response += "\n\nWould you like me to generate your itinerary now, or would you prefer to add more details?"
                        st.session_state.stage = "generating_itinerary"
                    else:
                        # Generate follow-up questions for missing information
                        follow_ups = follow_up_chain.run(
                            current_info=format_travel_info_summary(st.session_state.travel_info),
                            chat_history=st.session_state.chat_history
                        )
                        
                        if missing_details:
                            response += "\n\n**I still need the following information to create your itinerary:**\n"
                            for detail in missing_details:
                                response += f"- {detail.capitalize()}\n"
                        
                        response += "\n\n" + follow_ups
                
                elif st.session_state.stage == "generating_itinerary":
                    if "generate" in prompt.lower() or "create" in prompt.lower() or "yes" in prompt.lower() or "now" in prompt.lower():
                        # Generate the itinerary
                        with st.spinner("Creating your personalized itinerary..."):
                            itinerary_text = itinerary_chain.run(
                                travel_info=format_travel_info_summary(st.session_state.travel_info),
                                chat_history=st.session_state.chat_history
                            )
                            
                            st.session_state.itinerary = itinerary_text
                            response = itinerary_text
                            st.session_state.stage = "refining_itinerary"
                    else:
                        # Extract any additional information
                        extracted_info_text = extract_info_chain.run(
                            input=prompt,
                            chat_history=st.session_state.chat_history
                        )
                        
                        # Parse the extracted information
                        extracted_info = parse_travel_info(extracted_info_text)
                        
                        # Update travel info with new information
                        st.session_state.travel_info = merge_travel_info(st.session_state.travel_info, extracted_info)
                        
                        response = f"I've updated your preferences! Here's what I know now:\n\n"
                        response += format_travel_info_summary(st.session_state.travel_info)
                        response += "\n\nReady to generate your itinerary now?"
                
                elif st.session_state.stage == "refining_itinerary":
                    # Handle itinerary refinement requests
                    if "regenerate" in prompt.lower() or "new itinerary" in prompt.lower() or "create again" in prompt.lower():
                        # Update preferences if needed
                        extracted_info_text = extract_info_chain.run(
                            input=prompt,
                            chat_history=st.session_state.chat_history
                        )
                        
                        # Parse the extracted information
                        extracted_info = parse_travel_info(extracted_info_text)
                        
                        # Update travel info with new information
                        st.session_state.travel_info = merge_travel_info(st.session_state.travel_info, extracted_info)
                        
                        # Regenerate the itinerary
                        with st.spinner("Regenerating your itinerary..."):
                            itinerary_text = itinerary_chain.run(
                                travel_info=format_travel_info_summary(st.session_state.travel_info),
                                chat_history=st.session_state.chat_history
                            )
                            
                            st.session_state.itinerary = itinerary_text
                            response = itinerary_text
                    
                    else:
                        # Use a simple LLM call for other refinement requests
                        refine_prompt = ChatPromptTemplate.from_messages([
                            SystemMessage(content="""You are a helpful travel assistant. The user has an existing itinerary and is asking for changes or information. 
                            Address their specific question or request about the itinerary."""),
                            MessagesPlaceholder(variable_name="chat_history"),
                            HumanMessage(content="{input}")
                        ])
                        
                        refine_chain = LLMChain(
                            llm=llm,
                            prompt=refine_prompt,
                            verbose=True
                        )
                        
                        response = refine_chain.run(
                            input=prompt,
                            chat_history=st.session_state.chat_history
                        )
                
                # Display the response
                st.markdown(response)
                
                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history.append(AIMessage(content=response))
                
    # Show download button when itinerary is available
    if st.session_state.itinerary:
        st.sidebar.download_button(
            label="Download Itinerary",
            data=st.session_state.itinerary,
            file_name=f"{st.session_state.travel_info.get('destination', 'Travel')}_Itinerary.md",
            mime="text/markdown"
        )

if __name__ == "__main__":
    main()