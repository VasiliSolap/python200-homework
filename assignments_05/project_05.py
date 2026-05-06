from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


# Task 1: Setup and System Prompt

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

SYSTEM_PROMPT = """You are a job application coach. Your role is to help me create, refine, and optimize 
job application materials, including resumes, cover letters, LinkedIn profiles, 
and interview answers. Stay focused only on job application–related tasks.
Provide clear, practical, and actionable suggestions. When appropriate, rewrite or improve my 
content while preserving my intent and experience.
Always remind me to carefully review and edit any output before submitting it anywhere.
Acknowledge that you may not fully understand the specific norms, expectations, or standards 
of my industry, so I should use my own judgment and adapt the advice accordingly.
Ask clarifying questions when needed, but avoid unnecessary questions.
Be honest and constructive—if something is weak or unclear, point it out and suggest improvements."""


# Task 2: Bullet Point Rewriter

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    response = get_completion(messages, temperature=0.3) 

    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        result = json.loads(cleaned)
        for item in result:
            print(f"Original: {item['original']}")
            print(f"Improved: {item['improved']}")
            print()
        return result
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")
        print(response)
        return []
# Test
bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

print("--- Task 2: Bullet Point Rewriter ---")
rewrite_bullets(bullets)


#Task 3: Cover Letter Generator

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    return get_completion(messages, temperature=0.7)

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

print("--- Task 3: Cover Letter Generator ---")
result = generate_cover_letter(job_title, background)
print(result)

# The few-shot pattern controls tone — keeping it confident and specific
# rather than generic. Without examples, the model tends to produce
# clichés

# Task 4: Moderation Check

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged

    if flagged:
        print("I'm sorry, but I can't process that request.")
        print("Please rephrase your message and try again.")
    return not flagged

# Test
print("--- Task 4: Moderation Check ---")
safe_input = "Can you help me rewrite my resume bullet points?"
unsafe_input = "I want to dance in fire."

print(f"Test 1 (safe): {is_safe(safe_input)}")
print(f"Test 2 (unsafe): {is_safe(unsafe_input)}")


# Task 5: Chatbot Loop

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            rewrite_bullets(raw_bullets)
            print("Remember to review and edit these before submitting!\n")
            # Add to history so bot remembers context
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": "I rewrote your resume bullet points above."})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            result = generate_cover_letter(job_title, background)
            print(f"\nJob Application Helper:\n{result}\n")
            print("Remember to review and edit this before submitting!\n")
            # Add to history so bot remembers context
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": "I generated a cover letter opening above."})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"\nJob Application Helper: {reply}\n")
            messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    run_chatbot()


# Task 6: Ethics Reflection

# Question 1: Bias in the model's advice
# The bot was trained on a lot of text from certain industries like tech and finance.
# This means it might give better advice for people applying to those jobs.
# If someone is applying for a job in a different field, like construction or healthcare,
# the advice might not fit well. Also, the model uses a very confident and direct style
# of writing. This might not feel natural for people from cultures where being humble
# is more normal in professional settings. So the bot could accidentally favor certain
# types of people over others.

# Question 2: What could go wrong without review
# If someone sends the bot's output directly to an employer without checking it first,
# a few things could go wrong. The model sometimes adds numbers like "increased sales by 20%"
# that the user never mentioned. This would be dishonest on a real resume.
# Also, many employers can now recognize AI-generated text, and this could hurt
# the applicant's chances. The cover letter might also sound too generic and not
# reflect the real personality of the person applying.

# Question 3: One guardrail for professional deployment
# If I was going to use this tool for real, I would add a warning message
# that shows up after every response from the bot.
# The message would say something like:
# "Please check everything before you send it to an employer.
# Make sure all the facts and numbers are true and come from your real experience."
# This is important because the bot can sometimes invent details that sound good
# but are not true. A simple warning can remind the user to be careful
# and take responsibility for what they submit.