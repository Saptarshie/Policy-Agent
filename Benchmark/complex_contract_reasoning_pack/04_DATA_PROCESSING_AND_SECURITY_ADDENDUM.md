# Data Processing and Security Addendum

**Addendum ID:** DPA-AB-2026  
**Controller:** BlueHarbor Retail Limited  
**Processor:** Alderstone Analytics Private Limited  
**Effective Date:** 1 February 2026  
**Related Agreement:** MSA-AB-2026 and Order Form OF-7

> This is a fictional addendum for evaluating legal-document reasoning. It is not a representation of any actual law or compliance framework.

## 1. Scope and Priority

This DPA applies when Provider processes Personal Data for Customer in providing the services. For privacy, information security, processing, breach, audit, deletion, subprocessor, and transfer matters, this DPA prevails over inconsistent provisions in the MSA, SLS, or Order Form. A Change Order overrides this DPA only if it expressly names the DPA section being amended.

## 2. Defined Terms

- **Personal Data:** Information relating to an identified or reasonably identifiable natural person.
- **Customer Personal Data:** Personal Data processed by Provider on Customer's documented instructions.
- **Security Incident:** Unauthorized access to, acquisition, disclosure, alteration, loss, or destruction of Customer Personal Data. Unsuccessful attempts that do not compromise data, including blocked scans and failed login attempts, are not Security Incidents.
- **Subprocessor:** A third party engaged by Provider to process Customer Personal Data.
- **Restricted Data:** Government identifiers, payment-card data, precise geolocation, biometric templates, health information, and authentication secrets.

## 3. Processing Instructions

Provider will process Customer Personal Data only to provide, secure, maintain, support, and improve the contracted services, and as required by law. Provider must inform Customer if it believes an instruction violates applicable data-protection law, unless prohibited.

Provider may not use Customer Personal Data for advertising, sale, data-broker activity, or training a general-purpose model shared across unrelated customers. Provider may use de-identified telemetry to improve reliability if the data cannot reasonably be reidentified and Provider contractually prohibits reidentification.

## 4. Data Categories and Retention

Expected data includes employee identifiers, store-user activity, supplier contacts, product and transaction records that may be linked to loyalty identifiers, and support communications. Restricted Data is not authorized unless an executed Change Order expressly permits the specific category and states additional safeguards.

During the term, Customer controls configured retention where the service supports it. Provider may retain operational backups for up to 35 days after deletion from active systems. Security logs may be retained for 400 days. Legally required records may be retained only for the required period and must remain protected.

## 5. Security Measures

Provider will maintain:

- Encryption in transit using current industry-standard protocols.
- Encryption at rest for production Customer Personal Data.
- Multi-factor authentication for privileged administrative access.
- Least-privilege access reviews at least quarterly.
- Logging of privileged access and material security events.
- Annual independent penetration testing and remediation based on risk.
- Secure development, vulnerability management, backups, incident response, and business continuity controls.
- Background screening where lawful for personnel with privileged production access.

Provider must remediate critical vulnerabilities in internet-facing production systems within 15 calendar days after confirmation, unless a documented compensating control reduces the risk and Customer's security contact is notified.

## 6. Security Incidents

### 6.1 Notification

Provider must notify Customer without undue delay and no later than 24 hours after confirming a Security Incident. An initial notice may be preliminary and must not be delayed solely to obtain complete information.

### 6.2 Contents and Updates

Notice will describe, to the extent known, the nature of the incident, affected data and individuals, likely consequences, containment, and a contact point. Provider will provide material updates at least every 24 hours during active containment and a root-cause report within ten business days after containment, unless law enforcement requests delay.

### 6.3 Costs

Provider will bear reasonable, documented costs of investigation, legally required notifications, call-center support, and identity protection to the extent the Security Incident resulted from Provider's breach of this DPA. These costs are direct damages and are subject to the MSA Enhanced Cap, except where the claim arises from Provider's fraud or willful misconduct.

### 6.4 No Admission

Incident notification is not an admission of fault. Customer controls legally required notices to individuals and regulators, but Provider may make a notice required by law after consulting Customer where practical.

## 7. Subprocessors

The approved Subprocessor List is maintained in Provider's trust portal. Provider will give at least 30 calendar days' advance notice of a new Subprocessor that will process Customer Personal Data.

Customer may object within 15 calendar days after notice on reasonable, documented data-protection grounds. The parties will work in good faith for 15 days to resolve the objection. If no commercially reasonable solution is available, Customer may terminate only the affected service by notice within the following 15 days and receive a pro-rata refund of prepaid unused fees. Failure to object within the first 15 days constitutes approval of that Subprocessor.

Provider remains responsible for Subprocessor compliance to the same extent as for its own obligations.

## 8. Cross-Border Transfers and Data Location

Primary production Customer Personal Data will be hosted in India. Provider may allow remote administrative access from Singapore and the United Kingdom for support if access is logged, least-privileged, encrypted, and subject to a legally valid transfer mechanism where required.

Provider may move primary production hosting outside India only with Customer's prior written consent through a signed Change Order expressly amending this Section 8. Disaster-recovery copies may be stored in Singapore, encrypted, with access limited to recovery testing or an actual continuity event.

## 9. Individual Rights and Government Requests

Provider will reasonably assist Customer with verified requests relating to access, correction, deletion, restriction, or portability. Unless legally prohibited, Provider must redirect government requests for Customer Personal Data to Customer. If compelled to respond, Provider will give notice and disclose only the required data.

## 10. Audits

Provider will annually provide an independent control report or equivalent security assessment. Customer may conduct one remote audit per year with 20 business days' notice. An onsite audit is permitted only after a material Security Incident attributable to Provider, where a regulator requires it, or where the supplied report leaves a material control concern unresolved.

Audits must avoid unreasonable disruption, protect other customers' information, and be conducted by personnel bound by confidentiality. Customer bears ordinary audit costs. Provider bears incremental reasonable audit costs if the audit identifies a material uncured breach of this DPA.

## 11. Return and Deletion

During the term, Customer may export data using standard tools. On termination, Provider will keep Customer Data available for export for 30 calendar days unless Customer requests earlier deletion. After that period, Provider will delete active copies within 15 days and allow backups to expire within 35 days, except legally required retention.

If Customer terminates because of a confirmed Security Incident caused by Provider's willful misconduct, Provider will provide reasonable transition exports for up to 60 days without additional export fees. Ordinary professional services remain chargeable unless necessary to extract data in a documented standard format.

## 12. Restricted Data Exception Procedure

Customer must not upload Restricted Data by relying on an email, support ticket, or verbal assurance. Authorization requires a signed Change Order that:

1. Names each authorized Restricted Data category.
2. Describes the business purpose.
3. Adds appropriate technical and organizational safeguards.
4. States retention and deletion rules.
5. Expressly amends DPA Section 4.

If Customer uploads unauthorized Restricted Data, Provider must notify Customer when discovered and reasonably assist deletion. Customer bears resulting remediation costs unless Provider knowingly encouraged the upload contrary to this DPA.

## 13. Liability and Remedies

The MSA Enhanced Cap applies to ordinary DPA breaches and Provider-caused Security Incident costs. The MSA's fraud and willful-misconduct exclusions remain uncapped. Service credits do not reduce damages for a Security Incident unless Customer's damages claim includes the same specific loss compensated by the credit.

A breach of this DPA may also be a material breach of the affected Order Form if it creates material risk and is not cured within the MSA cure period. A Security Incident alone does not automatically establish breach, because fault depends on compliance with this DPA.

## 14. Survival

This DPA survives while Provider retains Customer Personal Data. Deletion, confidentiality, audit cooperation relating to pre-termination events, and liability provisions survive as necessary to give them effect.
