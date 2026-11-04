# Service Level and Premium Support Schedule

**Schedule ID:** SLS-AB-2026  
**Applies to:** Order Form OF-7  
**Effective Date:** 1 February 2026  
**Measurement Time Zone:** India Standard Time

> This fictional schedule contains interacting definitions, exclusions, credit formulas, escalation paths, and chronic-failure remedies.

## 1. Covered Production Service

The Covered Service includes the production Enterprise Analytics Cloud, Demand Forecasting Module, and Inventory Optimization Module. Nonproduction environments, beta functionality, customer-built code, optional third-party connectors, and professional services are not covered unless expressly stated.

## 2. Availability Commitment

Provider commits to **99.90% Monthly Availability** for the Covered Service.

Monthly Availability is calculated as:

`(Total Minutes - Excluded Minutes - Unavailable Minutes) / (Total Minutes - Excluded Minutes) × 100`

A minute is Unavailable when more than 20% of production users attempting access cannot authenticate or execute a core transaction because of a Provider-controlled failure.

## 3. Excluded Minutes

The following are excluded from both the numerator and denominator:

1. Scheduled Maintenance within the published window, up to four hours per calendar month.
2. Emergency Maintenance, up to two hours per month, if Provider gives as much notice as reasonably practical.
3. Customer systems, Customer Data defects, Customer-configured rules, or Customer network failures.
4. Third-party services selected and contracted directly by Customer.
5. Suspension permitted under the MSA.
6. Force majeure under the MSA.
7. A security isolation action reasonably necessary to contain a credible threat, but only if Provider follows the DPA and the action is not required because Provider failed to maintain an expressly required safeguard.

Scheduled Maintenance exceeding four hours remains excluded for the first four hours; excess minutes count as Unavailable if the Covered Service is unavailable. Emergency Maintenance exceeding two hours follows the same treatment.

## 4. Availability Credits

| Monthly Availability | Credit as % of monthly recurring fee for affected service |
|---|---:|
| 99.90% or above | 0% |
| 99.50% to 99.899% | 5% |
| 99.00% to 99.499% | 10% |
| Below 99.00% | 20% |

The monthly recurring fee is one-twelfth of the annual recurring fee allocated to the affected service. Premium Support fees are included only if Premium Support itself was materially unavailable.

Credits are Customer's sole monetary remedy for ordinary availability failure, but do not limit termination rights for Chronic Failure, claims arising from a security breach governed by the DPA, indemnity claims, or remedies that cannot lawfully be limited.

Credits for a month cannot exceed 20% of the monthly recurring fee for the affected service. Credits do not roll over beyond the next two invoices and are not cash refunds unless the affected Order Form terminates before application.

## 5. Credit Claim Process

Customer must submit a credit claim through the support portal within 30 calendar days after the month in which the incident occurred. The claim must identify dates, approximate times, affected locations, and ticket numbers where available.

Provider's monitoring data is presumed accurate, but Customer may rebut it with reasonably reliable logs. Provider must not reject a claim solely because Customer lacks a ticket number if Provider's status page or monitoring confirms the incident.

## 6. Incident Priorities

### Priority 1, Critical

Production service is unavailable to more than 20% of users, there is confirmed unauthorized access to Customer Data, or a critical business process has no reasonable workaround.

- Initial response: 15 minutes, 24×7.
- Status updates: every 30 minutes.
- Target workaround: 4 hours.
- Root-cause report: within 5 business days after resolution.

### Priority 2, High

Severe degradation or major functionality unavailable, but a reasonable workaround exists.

- Initial response: 1 hour, 24×7.
- Status updates: every 2 hours during active work.
- Target workaround: 8 hours.

### Priority 3, Normal

Limited impact, noncritical defect, or general service request.

- Initial response: 8 business hours.
- Updates: every 3 business days while open.

Response and workaround times are targets, not independent service-credit commitments, except as provided by Section 7.

## 7. Support Responsiveness Credits

If Provider misses the Priority 1 initial-response target by more than 15 minutes, Customer earns a credit equal to 2% of the monthly Premium Support fee for each qualifying incident, capped at 10% of that monthly fee per month.

No responsiveness credit applies if Customer incorrectly classifies an incident as Priority 1 and the facts reasonably support a lower priority. Provider must not downgrade an incident solely to avoid a target; the actual impact controls.

Availability and responsiveness credits may both apply to the same incident, but each is calculated against its own fee base and cap.

## 8. Chronic Failure

A Chronic Failure occurs if any one of the following happens:

1. Monthly Availability is below 99.00% in any three months within a rolling six-month period.
2. Monthly Availability is below 98.00% in any single month.
3. Five or more qualifying Priority 1 incidents occur within any rolling 90-day period, excluding incidents caused by Customer or an excluded event.

After a Chronic Failure, Customer may terminate the affected service by legal notice delivered within 30 calendar days after the condition is met. Termination is effective on the date stated by Customer, which must be between 15 and 60 days after notice. Provider must refund prepaid unused recurring fees for the terminated service.

If the base Enterprise Analytics Cloud is terminated for Chronic Failure, Customer may also terminate dependent modules. If only a module suffers Chronic Failure, Customer may terminate that module but not the unaffected base platform.

## 9. Planned Maintenance

The standard maintenance window is Sunday 00:00–04:00 IST. Provider will give at least five calendar days' notice for Scheduled Maintenance. Maintenance outside this window is not Scheduled Maintenance unless Customer approves it in writing.

Customer approval of an out-of-window maintenance event makes it Scheduled Maintenance for exclusion purposes but does not increase the four-hour monthly exclusion cap.

## 10. Disaster Recovery

Provider will maintain a recovery time objective of four hours and recovery point objective of one hour for a declared regional disaster affecting production. These are architecture objectives, not guaranteed restoration times. Failure to meet them counts toward availability if the service is Unavailable and is not otherwise excluded.

Provider will conduct an annual disaster-recovery test and provide a summary on request. Customer may observe one test per year remotely with 30 days' notice.

## 11. Premium Support Access

Customer may designate up to 25 Authorized Support Contacts. Emergency calls from other Customer personnel will be accepted for a suspected Priority 1 incident, but Customer must associate the incident with an Authorized Support Contact within one business day.

## 12. Service Measurement Across Modules

If the base platform is unavailable, each dependent module is considered unavailable for measurement, but Customer may not receive duplicative availability credits on both the base platform and a module for the same user impact. The credit is calculated on the highest applicable monthly recurring fee among the affected services.

If a module alone is unavailable while the base platform functions, the credit applies to that module's allocated monthly fee.

## 13. Reporting and Audit

Provider will deliver a monthly availability report within ten business days after month-end. Once per contract year, Customer may request reasonable evidence supporting calculations. Audit access to security logs or personal data is governed by the DPA.

## 14. Exceptions Do Not Rewrite the Agreement

A one-time operational concession, courtesy credit, priority upgrade, or extended support effort does not amend this SLS or create a precedent. Amendments require a signed document with authority under the MSA.
