import json

from ticket_router_base.config import QUEUES, PRIORITIES


FEW_SHOT_EXAMPLES = [
    {
        "queue": "Billing and Payments",
        "priority": "high",
        "tags": ["Billing Issue", "Invoice Review"],
        "preliminary_answer": "Dear customer, we have received your billing inquiry and are reviewing the charges on your account. We will provide a detailed breakdown within 24 hours. Thank you for your patience.",
    },
    {
        "queue": "Technical Support",
        "priority": "high",
        "tags": ["Network Hardware", "Router Issue"],
        "preliminary_answer": "Sehr geehrter Kunde, wir haben Ihre Meldung zu den Netzwerkproblemen mit Ihrem Cisco Router erhalten. Unser technisches Team wird sich schnellstmöglich mit Ihnen in Verbindung setzen.",
    },
    {
        "queue": "Customer Service",
        "priority": "low",
        "tags": ["Account Assistance", "General Inquiry"],
        "preliminary_answer": "Olá, agradecemos o seu contacto. Estamos a rever a sua solicitação e responderemos em breve com as informações necessárias.",
    },
    {
        "queue": "Product Support",
        "priority": "medium",
        "tags": ["Hardware Issue", "Device Troubleshooting"],
        "preliminary_answer": "Bonjour, nous avons bien reçu votre signalement concernant les problèmes techniques avec votre appareil. Notre équipe de support produit analysera votre dossier et reviendra vers vous sous 48 heures.",
    },
    {
        "queue": "IT Support",
        "priority": "medium",
        "tags": ["Cloud Infrastructure", "Performance Optimization"],
        "preliminary_answer": "Hello, thank you for reaching out regarding your cloud infrastructure needs. Our IT consulting team will prepare a customized optimization plan and contact you within 2 business days.",
    },
    {
        "queue": "Returns and Exchanges",
        "priority": "low",
        "tags": ["Product Return", "Order Issue"],
        "preliminary_answer": "Sehr geehrte Kundin, vielen Dank für Ihre Nachricht. Wir haben Ihre Rückgabeanfrage erhalten und werden Ihnen innerhalb von 3-5 Werktagen ein Rücksendeetikett zusenden.",
    },
    {
        "queue": "Service Outages and Maintenance",
        "priority": "high",
        "tags": ["Service Outage", "Incident Report"],
        "preliminary_answer": "Olá, lamentamos sinceramente pela interrupção do serviço. Nossa equipa de operações está a trabalhar ativamente. Fürneceremos uma atualização dentro de 60 minutos.",
    },
    {
        "queue": "Sales and Pre-Sales",
        "priority": "low",
        "tags": ["Feature Request", "Software Integration"],
        "preliminary_answer": "Estimado cliente, gracias por su interés en nuestra solución. Nuestro equipo de preventas analizará su solicitud y se pondrá en contacto con usted en breve para coordinar una demostración personalizada.",
    },
    {
        "queue": "Human Resources",
        "priority": "medium",
        "tags": ["HR Inquiry", "Employee Support"],
        "preliminary_answer": "Dear employee, thank you for reaching out to HR. We have received your inquiry and will follow up with the relevant department. Please expect a response within 3 business days.",
    },
    {
        "queue": "General Inquiry",
        "priority": "low",
        "tags": ["General Information", "Customer Inquiry"],
        "preliminary_answer": "Estimado cliente, gracias por ponerse en contacto con nosotros. Hemos registrado su consulta y le proporcionaremos la información solicitada en breve. Atentamente, el equipo de atención al cliente.",
    },
]


def build_system_prompt(few_shot: bool = True) -> str:
    base = (
        "You are a multilingual customer support assistant. "
        "Analyze the user's email (subject and body) and respond ONLY with a valid JSON object. "
        "Do not include markdown formatting outside the JSON.\n\n"
        "The JSON must have these exact keys:\n"
        "- queue: one of " + ", ".join(QUEUES) + "\n"
        "- priority: one of " + ", ".join(PRIORITIES) + "\n"
        "- tags: an array of 1-2 relevant tags (strings)\n"
        "- preliminary_answer: a polite, helpful customer service reply in the same language as the user's email\n"
    )
    if not few_shot:
        return base + (
            "\nExample:\n"
            '{"queue": "Billing and Payments", "priority": "high", "tags": ["Billing Issue"], "preliminary_answer": "Dear customer, we are reviewing your charges and will respond within 24 hours."}'
        )
    examples_str = "\n".join(
        '{{"queue": "{q}", "priority": "{p}", "tags": {t}, "preliminary_answer": "{a}"}}'.format(
            q=e["queue"],
            p=e["priority"],
            t=json.dumps(e["tags"]),
            a=e["preliminary_answer"],
        )
        for e in FEW_SHOT_EXAMPLES
    )
    return base + "\nExamples:\n" + examples_str


def build_prompt(
    subject: str, body: str, language: str | None, few_shot: bool = True
) -> str:
    system = build_system_prompt(few_shot=few_shot)
    user_text = f"Subject: {subject}\nBody: {body}"

    if language:
        user_text = f"Language: {language}\n" + user_text

    return f"{system}\n\n{user_text}"
