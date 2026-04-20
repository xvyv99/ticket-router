from ticket_router_base.config import QUEUES, PRIORITIES
from ticket_router_base.types import GroundRecord, Queue, Priority

FEW_SHOT_EXAMPLES = [
    GroundRecord(
        queue=Queue.BILLING_AND_PAYMENTS,
        priority=Priority.HIGH,
        tag_1="Billing Issue",
        tag_2="Invoice Review",
        answer="Dear customer, we have received your billing inquiry and are reviewing the charges on your account. We will provide a detailed breakdown within 24 hours. Thank you for your patience.",
    ),
    GroundRecord(
        queue=Queue.TECHNICAL_SUPPORT,
        priority=Priority.HIGH,
        tag_1="Network Hardware",
        tag_2="Router Issue",
        answer="Sehr geehrter Kunde, wir haben Ihre Meldung zu den Netzwerkproblemen mit Ihrem Cisco Router erhalten. Unser technisches Team wird sich schnellstmöglich mit Ihnen in Verbindung setzen.",
    ),
    GroundRecord(
        queue=Queue.CUSTOMER_SERVICE,
        priority=Priority.LOW,
        tag_1="Account Assistance",
        tag_2="General Inquiry",
        answer="Olá, agradecemos o seu contacto. Estamos a rever a sua solicitação e responderemos em breve com as informações necessárias.",
    ),
    GroundRecord(
        queue=Queue.PRODUCT_SUPPORT,
        priority=Priority.MEDIUM,
        tag_1="Hardware Issue",
        tag_2="Device Troubleshooting",
        answer="Bonjour, nous avons bien reçu votre signalement concernant les problèmes techniques avec votre appareil. Notre équipe de support produit analysera votre dossier et reviendra vers vous sous 48 heures.",
    ),
    GroundRecord(
        queue=Queue.IT_SUPPORT,
        priority=Priority.MEDIUM,
        tag_1="Cloud Infrastructure",
        tag_2="Performance Optimization",
        answer="Hello, thank you for reaching out regarding your cloud infrastructure needs. Our IT consulting team will prepare a customized optimization plan and contact you within 2 business days.",
    ),
    GroundRecord(
        queue=Queue.RETURNS_AND_EXCHANGES,
        priority=Priority.LOW,
        tag_1="Product Return",
        tag_2="Order Issue",
        answer="Sehr geehrte Kundin, vielen Dank für Ihre Nachricht. Wir haben Ihre Rückgabeanfrage erhalten und werden Ihnen innerhalb von 3-5 Werktagen ein Rücksendeetikett zusenden.",
    ),
    GroundRecord(
        queue=Queue.SERVICE_OUTAGES_AND_MAINTENANCE,
        priority=Priority.HIGH,
        tag_1="Service Outage",
        tag_2="Incident Report",
        answer="Olá, lamentamos sinceramente pela interrupção do serviço. Nossa equipa de operações está a trabalhar ativamente. Fürneceremos uma atualização dentro de 60 minutos.",
    ),
    GroundRecord(
        queue=Queue.SALES_AND_PRE_SALES,
        priority=Priority.LOW,
        tag_1="Feature Request",
        tag_2="Software Integration",
        answer="Estimado cliente, gracias por su interés en nuestra solución. Nuestro equipo de preventas analizará su solicitud y se pondrá en contacto con usted en breve para coordinar una demostración personalizada.",
    ),
    GroundRecord(
        queue=Queue.HUMAN_RESOURCES,
        priority=Priority.MEDIUM,
        tag_1="HR Inquiry",
        tag_2="Employee Support",
        answer="Dear employee, thank you for reaching out to HR. We have received your inquiry and will follow up with the relevant department. Please expect a response within 3 business days.",
    ),
    GroundRecord(
        queue=Queue.GENERAL_INQUIRY,
        priority=Priority.LOW,
        tag_1="General Information",
        tag_2="Customer Inquiry",
        answer="Estimado cliente, gracias por ponerse en contacto con nosotros. Hemos registrado su consulta y le proporcionaremos la información solicitada en breve. Atentamente, el equipo de atención al cliente.",
    ),
]

DEMO_RECORD = GroundRecord(
    queue=Queue.BILLING_AND_PAYMENTS,
    priority=Priority.HIGH,
    tag_1="Billing Issue",
    tag_2="Invoice Review",
    answer="Dear customer, we are reviewing your charges and will respond within 24 hours.",
)


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
        examples_str = DEMO_RECORD.to_json_str()
    else:
        examples_str = "\n".join(e.to_json_str() for e in FEW_SHOT_EXAMPLES)

    prompt = base + "\nExamples:\n" + examples_str

    return prompt


def build_prompt(
    subject: str, body: str, language: str | None, few_shot: bool = True
) -> str:
    system = build_system_prompt(few_shot=few_shot)
    user_text = f"Subject: {subject}\nBody: {body}"

    if language:
        user_text = f"Language: {language}\n" + user_text

    return f"{system}\n\n{user_text}"
