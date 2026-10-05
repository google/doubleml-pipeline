| Column Index | Column Name | Data Source | Observable | National | Geo | Week | Generation | Description | Note / Details |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Geo | Base data | Observable |  | Y |  |  | Unique identifier for the geographic area (1~30). | 30 geographic areas across Japan (Geo 1~30). |
| 2 | Week | Base data | Observable |  |  | Y |  | Weekly timestamp representing the start of the week. | Range: 3/1/2021 ~ 12/30/2024. |
| 3 | Holidays | Base data | Observable | Y |  | Y |  | Binary flag (1/0) indicating holidays assumed to impact sales. | Binary flag (1/0) indicating national or regional holidays. |
| 4 | DemandShock | Base data | Observable |  |  | Y |  | Binary flag (1/0) indicating exogenous demand shock events. | Binary flag (1/0) indicating exogenous demand shock events. |
| 5 | TotalDemandSeasonality | Base data | Observable | Y |  | Y |  | Seasonality index based on media planning assumptions. | Macro-level baseline demand seasonality index. |
| 6 | TotalDemand | Base data | Observable |  | Y | Y |  | Total market demand index for the period. | Total category demand index across all regions. |
| 7 | TotalGeoDemand | Generated data | Observable |  | Y | Y |  | Aggregated total demand across all categories for the specific Geo. |  |
| 8 | CostMultiplier_TV | Base data | Observable |  | Y | Y |  | Cost multiplier applied to the base TV unit cost for each Geo to represent regional cost differentials. | Regional cost multiplier relative to standard Geo unit cost. |
| 9 | Population | Base data | Observable |  | Y | Y |  | Total population count within the specific Geo. | Total demographic population count within the Geo. |
| 10 | Pop_Young | Base data | Observable |  | Y | Y | Y | Population count for specific generations (Young) in the Geo. | Population count for Young generation within the Geo. |
| 11 | Pop_Middle | Base data | Observable |  | Y | Y | Y | Population count for specific generations (Middle) in the Geo. | Population count for Middle generation within the Geo. |
| 12 | Pop_Elder | Base data | Observable |  | Y | Y | Y | Population count for specific generations (Elder) in the Geo. | Population count for Elder generation within the Geo. |
| 13 | ViewRate_Flyer_Young | Base data | Unobservable |  | Y | Y | Y | Conversion rate from Flyer distribution to viewing by generation (Young). |  |
| 14 | ViewRate_Flyer_Middle | Base data | Unobservable |  | Y | Y | Y | Conversion rate from Flyer distribution to viewing by generation (Middle). |  |
| 15 | ViewRate_Flyer_Elder | Base data | Unobservable |  | Y | Y | Y | Conversion rate from Flyer distribution to viewing by generation (Elder). |  |
| 16 | TotalDemandSeasonality(intentionaly) | Generated data | Observable |  | Y | Y |  | Geo-specific seasonality index with added noise for simulation variance. | Derived from 'TotalDemandSeasonality'. |
| 17 | Type_of_Plan_TV | Generated data | Observable |  | Y | Y |  | Categorical planning intensity scenario for TV ('NoSpend_Period', 'Low Intensity', 'Normal Period'). | Derived from seasonality thresholds. |
| 18 | Type_of_Plan_Digital | Generated data | Observable |  | Y | Y |  | Categorical planning intensity scenario for Digital ('NoSpend_Period', 'Low Intensity', 'Normal Period'). | Derived from seasonality thresholds. |
| 19 | Type_of_Plan_Flyer | Generated data | Observable |  | Y | Y |  | Categorical planning intensity scenario for Flyer ('NoSpend_Period', 'Low Intensity', 'Normal Period'). | Derived from seasonality thresholds. |
| 20 | Plan_TV | Generated data | Observable |  | Y | Y |  | Forecasted Target Reach determined during the budget planning phase for TV. | Baseline for buying reach. |
| 21 | Plan_Digital | Generated data | Observable |  | Y | Y |  | Forecasted Target Reach determined during the budget planning phase for Digital. | Baseline for buying reach. |
| 22 | Plan_Flyer | Generated data | Observable |  | Y | Y |  | Forecasted Target Reach determined during the budget planning phase for Flyer. | Baseline for buying reach. |
| 23 | Executed_TV | Generated data | Observable |  | Y | Y |  | Executed reach metric containing simulated variance from the Plan for TV. | Simulated execution reach for TV. |
| 24 | Executed_Digital | Generated data | Observable |  | Y | Y |  | Executed reach metric containing simulated variance from the Plan for Digital. | Simulated execution reach for Digital. |
| 25 | Executed_Flyer | Generated data | Observable |  | Y | Y |  | Executed reach metric containing simulated variance from the Plan for Flyer. | Simulated execution reach for Flyer. |
| 26 | UnitCost_TV | Base data | Observable |  | Y | Y |  | Cost per unit (e.g., GRP, impression) for TV. | Base media cost per unit (GRP) for TV by Geo. |
| 27 | UnitCost_Flyer | Base data | Observable |  | Y | Y |  | Cost per unit (e.g., GRP, impression) for Flyer. | Base media cost per unit (distribution count) for Flyer by Geo. |
| 28 | Impression_TV | Generated data | Observable |  | Y | Y |  | Total impressions delivered for TV. |  |
| 29 | Impression_Digital | Generated data | Observable |  | Y | Y |  | Total impressions delivered for Digital. |  |
| 30 | Impression_Flyer | Generated data | Observable |  | Y | Y |  | Total impressions delivered for Flyer. |  |
| 31 | National_Population | Generated data | Observable | Y |  | Y |  | Aggregated national population (Sum of all Geo populations). |  |
| 32 | Spend_TV | Generated data | Observable |  | Y | Y |  | Total advertising spend (JPY) for TV. |  |
| 33 | Spend_Digital | Generated data | Observable |  | Y | Y |  | Total advertising spend (JPY) for Digital. |  |
| 34 | Spend_Flyer | Generated data | Observable |  | Y | Y |  | Total advertising spend (JPY) for Flyer. |  |
| 35 | Impression_TV_Young | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Young) for TV. |  |
| 36 | Impression_TV_Middle | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Middle) for TV. |  |
| 37 | Impression_TV_Elder | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Elder) for TV. |  |
| 38 | Impression_Digital_Young | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Young) for Digital. |  |
| 39 | Impression_Digital_Middle | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Middle) for Digital. |  |
| 40 | Impression_Digital_Elder | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Elder) for Digital. |  |
| 41 | Impression_Flyer_Young | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Young) for Flyer. |  |
| 42 | Impression_Flyer_Middle | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Middle) for Flyer. |  |
| 43 | Impression_Flyer_Elder | Generated data | Observable |  | Y | Y | Y | Impressions delivered to a specific generation segment (Elder) for Flyer. |  |
| 44 | Frequency_TV | Generated data | Observable |  | Y | Y |  | Average frequency of ad exposure for TV. |  |
| 45 | Frequency_Digital | Generated data | Observable |  | Y | Y |  | Average frequency of ad exposure for Digital. |  |
| 46 | Frequency_Flyer | Generated data | Observable |  | Y | Y |  | Average frequency of ad exposure for Flyer. |  |
| 47 | Reach_TV_Young | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Young) for TV. |  |
| 48 | Reach_TV_Middle | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Middle) for TV. |  |
| 49 | Reach_TV_Elder | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Elder) for TV. |  |
| 50 | Reach_Digital_Young | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Young) for Digital. |  |
| 51 | Reach_Digital_Middle | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Middle) for Digital. |  |
| 52 | Reach_Digital_Elder | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Elder) for Digital. |  |
| 53 | Reach_Flyer_Young | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Young) for Flyer. |  |
| 54 | Reach_Flyer_Middle | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Middle) for Flyer. |  |
| 55 | Reach_Flyer_Elder | Generated data | Observable |  | Y | Y | Y | Reach count targeting a specific generation segment (Elder) for Flyer. |  |
| 56 | Reach_Flyer_Young_L1 | Generated data | Unobservable |  | Y | Y | Y | 1-week Lagged Reach: Delayed reach effect carrying over from 1 week(s) prior (Adstock) for Young. | L1 = 1 week(s) lag. |
| 57 | Reach_Flyer_Middle_L1 | Generated data | Unobservable |  | Y | Y | Y | 1-week Lagged Reach: Delayed reach effect carrying over from 1 week(s) prior (Adstock) for Middle. | L1 = 1 week(s) lag. |
| 58 | Reach_Flyer_Elder_L1 | Generated data | Unobservable |  | Y | Y | Y | 1-week Lagged Reach: Delayed reach effect carrying over from 1 week(s) prior (Adstock) for Elder. | L1 = 1 week(s) lag. |
| 59 | Reach_Flyer_Young_L2 | Generated data | Unobservable |  | Y | Y | Y | 2-week Lagged Reach: Delayed reach effect carrying over from 2 week(s) prior (Adstock) for Young. | L2 = 2 week(s) lag. |
| 60 | Reach_Flyer_Middle_L2 | Generated data | Unobservable |  | Y | Y | Y | 2-week Lagged Reach: Delayed reach effect carrying over from 2 week(s) prior (Adstock) for Middle. | L2 = 2 week(s) lag. |
| 61 | Reach_Flyer_Elder_L2 | Generated data | Unobservable |  | Y | Y | Y | 2-week Lagged Reach: Delayed reach effect carrying over from 2 week(s) prior (Adstock) for Elder. | L2 = 2 week(s) lag. |
| 62 | Reach_Flyer_Young_L3 | Generated data | Unobservable |  | Y | Y | Y | 3-week Lagged Reach: Delayed reach effect carrying over from 3 week(s) prior (Adstock) for Young. | L3 = 3 week(s) lag. |
| 63 | Reach_Flyer_Middle_L3 | Generated data | Unobservable |  | Y | Y | Y | 3-week Lagged Reach: Delayed reach effect carrying over from 3 week(s) prior (Adstock) for Middle. | L3 = 3 week(s) lag. |
| 64 | Reach_Flyer_Elder_L3 | Generated data | Unobservable |  | Y | Y | Y | 3-week Lagged Reach: Delayed reach effect carrying over from 3 week(s) prior (Adstock) for Elder. | L3 = 3 week(s) lag. |
| 65 | View_Flyer_Young_Directly | Generated data | Unobservable |  | Y | Y | Y | Flyer views generated directly without influence from other media for Young. |  |
| 66 | View_Flyer_Middle_Directly | Generated data | Unobservable |  | Y | Y | Y | Flyer views generated directly without influence from other media for Middle. |  |
| 67 | View_Flyer_Elder_Directly | Generated data | Unobservable |  | Y | Y | Y | Flyer views generated directly without influence from other media for Elder. |  |
| 68 | View_Flyer_Young_viaTV | Generated data | Unobservable |  | Y | Y | Y | Flyer views induced/triggered by TV ad exposure (Synergy effect) for Young. |  |
| 69 | View_Flyer_Middle_viaTV | Generated data | Unobservable |  | Y | Y | Y | Flyer views induced/triggered by TV ad exposure (Synergy effect) for Middle. |  |
| 70 | View_Flyer_Elder_viaTV | Generated data | Unobservable |  | Y | Y | Y | Flyer views induced/triggered by TV ad exposure (Synergy effect) for Elder. |  |
| 71 | View_Flyer_Young_viaDigital | Generated data | Unobservable |  | Y | Y | Y | Flyer views induced/triggered by Digital ad exposure (Synergy effect) for Young. |  |
| 72 | View_Flyer_Middle_viaDigital | Generated data | Unobservable |  | Y | Y | Y | Flyer views induced/triggered by Digital ad exposure (Synergy effect) for Middle. |  |
| 73 | View_Flyer_Elder_viaDigital | Generated data | Unobservable |  | Y | Y | Y | Flyer views induced/triggered by Digital ad exposure (Synergy effect) for Elder. |  |
| 74 | View_Flyer_Young | Generated data | Unobservable |  | Y | Y | Y | Total Flyer views (Sum of Directly, viaTV, and viaDigital paths) for Young. |  |
| 75 | View_Flyer_Middle | Generated data | Unobservable |  | Y | Y | Y | Total Flyer views (Sum of Directly, viaTV, and viaDigital paths) for Middle. |  |
| 76 | View_Flyer_Elder | Generated data | Unobservable |  | Y | Y | Y | Total Flyer views (Sum of Directly, viaTV, and viaDigital paths) for Elder. |  |
| 77 | StoreCoverage_by_Geo | Base data | Observable |  | Y | Y |  | Store coverage / market presence of the total demand within the specific Geo. | Retailer's store coverage and market presence ratio by Geo. |
| 78 | Baseline | Generated data | Unobservable |  | Y | Y |  | Estimated baseline sales without marketing intervention. |  |
| 79 | ConversionRate_TV_Young | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Young (TV). |  |
| 80 | ConversionRate_TV_Middle | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Middle (TV). |  |
| 81 | ConversionRate_TV_Elder | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Elder (TV). |  |
| 82 | ConversionRate_Flyer_Young | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Young (Flyer). |  |
| 83 | ConversionRate_Flyer_Middle | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Middle (Flyer). |  |
| 84 | ConversionRate_Flyer_Elder | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Elder (Flyer). |  |
| 85 | ConversionRate_Digital_Young | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Young (Digital). |  |
| 86 | ConversionRate_Digital_Middle | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Middle (Digital). |  |
| 87 | ConversionRate_Digital_Elder | Base data | Unobservable |  | Y | Y | Y | Conversion rate from media exposure to purchase for Elder (Digital). |  |
| 88 | Sales_TV_Young | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to TV and Generation (Young). |  |
| 89 | Sales_TV_Middle | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to TV and Generation (Middle). |  |
| 90 | Sales_TV_Elder | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to TV and Generation (Elder). |  |
| 91 | Sales_Digital_Young | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Digital and Generation (Young). |  |
| 92 | Sales_Digital_Middle | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Digital and Generation (Middle). |  |
| 93 | Sales_Digital_Elder | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Digital and Generation (Elder). |  |
| 94 | Sales_Flyer_Directly_Young | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'Directly' for Young. | Influence path: Directly. |
| 95 | Sales_Flyer_Directly_Middle | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'Directly' for Middle. | Influence path: Directly. |
| 96 | Sales_Flyer_Directly_Elder | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'Directly' for Elder. | Influence path: Directly. |
| 97 | Sales_Flyer_viaTV_Young | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'viaTV' for Young. | Influence path: viaTV. |
| 98 | Sales_Flyer_viaTV_Middle | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'viaTV' for Middle. | Influence path: viaTV. |
| 99 | Sales_Flyer_viaTV_Elder | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'viaTV' for Elder. | Influence path: viaTV. |
| 100 | Sales_Flyer_viaDigital_Young | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'viaDigital' for Young. | Influence path: viaDigital. |
| 101 | Sales_Flyer_viaDigital_Middle | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'viaDigital' for Middle. | Influence path: viaDigital. |
| 102 | Sales_Flyer_viaDigital_Elder | Generated data | Unobservable |  | Y | Y | Y | Sales volume attributed to Flyer via path 'viaDigital' for Elder. | Influence path: viaDigital. |
| 103 | CouponRedemptionRate_TV_Young | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by TV for Young. |  |
| 104 | CouponRedemptionRate_TV_Middle | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by TV for Middle. |  |
| 105 | CouponRedemptionRate_TV_Elder | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by TV for Elder. |  |
| 106 | CouponRedemptionRate_Flyer_Young | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by Flyer for Young. |  |
| 107 | CouponRedemptionRate_Flyer_Middle | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by Flyer for Middle. |  |
| 108 | CouponRedemptionRate_Flyer_Elder | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by Flyer for Elder. |  |
| 109 | CouponRedemptionRate_Digital_Young | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by Digital for Young. |  |
| 110 | CouponRedemptionRate_Digital_Middle | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by Digital for Middle. |  |
| 111 | CouponRedemptionRate_Digital_Elder | Base data | Unobservable |  | Y | Y | Y | Rate of coupon redemption / usage response triggered by Digital for Elder. |  |
| 112 | CouponUsed_viaDigital_Young | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaDigital (Young). |  |
| 113 | CouponSales_viaDigital_Young | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaDigital (Young). |  |
| 114 | CouponDiscount_viaDigital_Young | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaDigital (Young). |  |
| 115 | CouponUsed_viaDigital_Middle | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaDigital (Middle). |  |
| 116 | CouponSales_viaDigital_Middle | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaDigital (Middle). |  |
| 117 | CouponDiscount_viaDigital_Middle | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaDigital (Middle). |  |
| 118 | CouponUsed_viaDigital_Elder | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaDigital (Elder). |  |
| 119 | CouponSales_viaDigital_Elder | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaDigital (Elder). |  |
| 120 | CouponDiscount_viaDigital_Elder | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaDigital (Elder). |  |
| 121 | CouponUsed_viaFlyer_Directly_Young | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (Directly_Young). |  |
| 122 | CouponSales_viaFlyer_Directly_Young | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (Directly_Young). |  |
| 123 | CouponDiscount_viaFlyer_Directly_Young | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (Directly_Young). |  |
| 124 | CouponUsed_viaFlyer_Directly_Middle | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (Directly_Middle). |  |
| 125 | CouponSales_viaFlyer_Directly_Middle | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (Directly_Middle). |  |
| 126 | CouponDiscount_viaFlyer_Directly_Middle | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (Directly_Middle). |  |
| 127 | CouponUsed_viaFlyer_Directly_Elder | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (Directly_Elder). |  |
| 128 | CouponSales_viaFlyer_Directly_Elder | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (Directly_Elder). |  |
| 129 | CouponDiscount_viaFlyer_Directly_Elder | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (Directly_Elder). |  |
| 130 | CouponUsed_viaFlyer_viaTV_Young | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (viaTV_Young). |  |
| 131 | CouponSales_viaFlyer_viaTV_Young | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (viaTV_Young). |  |
| 132 | CouponDiscount_viaFlyer_viaTV_Young | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (viaTV_Young). |  |
| 133 | CouponUsed_viaFlyer_viaTV_Middle | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (viaTV_Middle). |  |
| 134 | CouponSales_viaFlyer_viaTV_Middle | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (viaTV_Middle). |  |
| 135 | CouponDiscount_viaFlyer_viaTV_Middle | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (viaTV_Middle). |  |
| 136 | CouponUsed_viaFlyer_viaTV_Elder | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (viaTV_Elder). |  |
| 137 | CouponSales_viaFlyer_viaTV_Elder | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (viaTV_Elder). |  |
| 138 | CouponDiscount_viaFlyer_viaTV_Elder | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (viaTV_Elder). |  |
| 139 | CouponUsed_viaFlyer_viaDigital_Young | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (viaDigital_Young). |  |
| 140 | CouponSales_viaFlyer_viaDigital_Young | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (viaDigital_Young). |  |
| 141 | CouponDiscount_viaFlyer_viaDigital_Young | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (viaDigital_Young). |  |
| 142 | CouponUsed_viaFlyer_viaDigital_Middle | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (viaDigital_Middle). |  |
| 143 | CouponSales_viaFlyer_viaDigital_Middle | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (viaDigital_Middle). |  |
| 144 | CouponDiscount_viaFlyer_viaDigital_Middle | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (viaDigital_Middle). |  |
| 145 | CouponUsed_viaFlyer_viaDigital_Elder | Generated data | Observable |  | Y | Y |  | Number of coupons used through viaFlyer (viaDigital_Elder). |  |
| 146 | CouponSales_viaFlyer_viaDigital_Elder | Generated data | Observable |  | Y | Y |  | Sales volume generated specifically through coupon usage via viaFlyer (viaDigital_Elder). |  |
| 147 | CouponDiscount_viaFlyer_viaDigital_Elder | Generated data | Observable |  | Y | Y |  | Total discount amount (JPY) applied via coupons through viaFlyer (viaDigital_Elder). |  |
| 148 | CouponDiscount_viaDigital | Generated data | Observable |  | Y | Y |  | Aggregated coupon discount amount for the channel (viaDigital). |  |
| 149 | CouponDiscount_viaFlyer | Generated data | Observable |  | Y | Y |  | Aggregated coupon discount amount for the channel (viaFlyer). |  |
| 150 | CouponDiscount | Generated data | Observable |  | Y | Y |  | Total aggregated coupon discount amount across all channels. |  |
| 151 | a_Sales_TV_direct | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_TV_direct. | Decomposition component [a]. |
| 152 | b_Sales_TV_indirect_flyer_coupon | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_TV_indirect_flyer_coupon. | Decomposition component [b]. |
| 153 | c_Sales_TV_indirect_flyer_no_coupon | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_TV_indirect_flyer_no_coupon. | Decomposition component [c]. |
| 154 | d_Sales_Digital_direct | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_Digital_direct. | Decomposition component [d]. |
| 155 | e_Sales_Digital_indirect_coupon | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_Digital_indirect_coupon. | Decomposition component [e]. |
| 156 | f_Sales_Digital_indirect_flyer_coupon | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_Digital_indirect_flyer_coupon. | Decomposition component [f]. |
| 157 | g_Sales_Digital_indirect_flyer_no_coupon | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_Digital_indirect_flyer_no_coupon. | Decomposition component [g]. |
| 158 | h_Sales_Flyer_direct | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_Flyer_direct. | Decomposition component [h]. |
| 159 | i_Sales_Flyer_indirect_coupon | Generated data | Unobservable |  | Y | Y |  | Simulation Truth: Sales lift driven by Sales_Flyer_indirect_coupon. | Decomposition component [i]. |
| 160 | IncrementalSales | Generated data | Unobservable |  | Y | Y |  | Total Incremental Sales (Sum of components a through i). |  |
| 161 | Sales | Generated data | Observable |  | Y | Y |  | Total Sales (Sum of Baseline and Incremental Sales). |  |
