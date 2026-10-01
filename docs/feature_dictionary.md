# Feature definitions

Every predictor uses records strictly before the cutoff. Customer ID identifies a row and is never a predictor. Currency is GBP; time differences are fractional days.

| Feature | Definition and purpose |
|---|---|
| recency | Days since the last positive purchase; inactivity. |
| frequency | Distinct positive invoices over all observed history; purchase habit. |
| monetary | Historical gross positive spend; customer value. |
| average_spend | Mean positive invoice amount; typical basket value. |
| historical_clv | Signed historical net spend including returns; realized value proxy. |
| basket_size | Mean units per invoice; bulk versus small-basket behavior. |
| frequency_trend | Least-squares slope over six consecutive 30-day order-count bins, oldest to newest. |
| gap_variance | Sample variance of consecutive inter-order gaps; zero when undefined. |
| seasonal_concentration | Maximum calendar-month order count divided by all orders; seasonal concentration. |
| return_rate | Absolute returned units / purchased units; can exceed one if returns relate to unobserved purchases. |
| discount_proxy | Share of product lines below 90% of that SKU's pre-cutoff median price; not a measured promotion response. |
| engagement | 100/3 × [exp(-recency/90) + 1-exp(-frequency/10) + 1-exp(-monetary/2500)]. |
| tenure | Days since first observed purchase; observation exposure, not actual acquisition date. |
| orders_90d | Invoice count in the prior 90 days; recent activity. |
| spend_90d | Gross positive spend in the prior 90 days; recent value. |
| unique_products | Distinct product StockCodes; breadth of interests. |
| sku_popularity | Mean pre-cutoff catalog frequency of purchased SKUs; niche versus popular products. |
| affinity_home | Share matching HEART/CANDLE/LANTERN/FRAME/CLOCK/CUSHION. |
| affinity_kitchen | Share matching MUG/CUP/PLATE/BOWL/KITCHEN/TEA/CAKE after home precedence. |
| affinity_seasonal | Share matching CHRISTMAS/XMAS/EASTER/SANTA/HALLOWEEN after earlier rules. |
| affinity_accessories | Share matching BAG/NECKLACE/BRACELET/PURSE/RING after earlier rules. |
| affinity_other | Remaining product lines. These categories are description-based proxies. |
| country | Last observed purchasing country; one-hot encoding, unseen country ignored at inference. |

StockCodes matching `^\d{5}[A-Z]*$` are treated as products. Administrative codes remain in accounting/RFM sums but are excluded from product affinity/recommendations. This is a documented heuristic, not a perfect merchant product master. A modal normalized description per SKU reconciles inconsistent wording, fitted only on pre-cutoff history.

Catalog popularity and price references are shared pre-cutoff observations across customers. This represents a known-catalog historical snapshot, not an unseen-catalog experiment. Numeric imputation and model scaling remain training-only within every CV fold.

The LSTM uses the last 12 product invoices, with log(1+amount)/10, log(1+gap)/7, log(1+age at cutoff)/7 and five category shares. Packed lengths ignore trailing padding. A customer without product-coded invoices receives one all-zero event.

High/infinite VIF can occur because monetary/net spend, RFM composites and affinity shares are redundant. VIF is explicitly exported; these PRD feature families are retained for tree models and regularized baselines. Coefficients are not causal estimates.
