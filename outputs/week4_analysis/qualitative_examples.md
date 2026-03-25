# Week 5 Qualitative Analysis Notes

The examples below highlight where agreement-gated reranking helps and where it can still fail.

## xsum

### xsum Example 290 (improvement)
- Reference: Manchester United striker Zlatan Ibrahimovic says "nothing is done" with regards to his future but insists he has fulfilled requirements needed to extend his contract.
- Top-1: Manchester United striker Zlatan Ibrahimovic says he will "wait and see" if he will remain at the club next season.
- Agreement-gated: Manchester United striker Zlatan Ibrahimovic says he is yet to decide his future at the club.
- Top-1 scores: FactCC=0.0132, NLI=0.1929, SummaC=0.2112
- Agreement-gated scores: FactCC=0.9846, NLI=0.9400, SummaC=0.2084
- Gate reason: metric_agreement

### xsum Example 126 (improvement)
- Reference: Only a day after it opened, Iran has shut down what officials reportedly thought was a branch of the US fast food giant Kentucky Fried Chicken.
- Top-1: Police in the Iranian capital, Tehran, have shut down a branch of the fast food chain KFC, state media report.
- Agreement-gated: Police in the Iranian capital, Tehran, have shut down a branch of the fast food chain KFC.
- Top-1 scores: FactCC=0.2857, NLI=0.0232, SummaC=0.2158
- Agreement-gated scores: FactCC=0.9832, NLI=0.9136, SummaC=0.2158
- Gate reason: metric_agreement

### xsum Example 140 (improvement)
- Reference: Home webcams that were hijacked to help knock popular websites offline last week are being recalled in the US.
- Top-1: A company that makes webcams has recalled all of its products after they were used in a massive web attack last month.
- Agreement-gated: A company that makes webcams has recalled all of its products after they were used in a large web attack.
- Top-1 scores: FactCC=0.0088, NLI=0.0100, SummaC=0.2501
- Agreement-gated scores: FactCC=0.9610, NLI=0.4915, SummaC=0.2532
- Gate reason: metric_agreement

### xsum Example 67 (failure)
- Reference: A convicted teenage sex offender has been detained for a year for having sex with a 13-year-old girl in the grounds of a Dundee college campus.
- Top-1: A teenager who had sex with a 13-year-old girl in the grounds of a Dundee college has been detained for one year.
- Agreement-gated: A Dundee teenager who had sex with a 13-year-old girl after meeting her on Snapchat has been detained for one year.
- Top-1 scores: FactCC=0.9842, NLI=0.0029, SummaC=0.2246
- Agreement-gated scores: FactCC=0.0185, NLI=0.0039, SummaC=0.2275
- Gate reason: metric_agreement

### xsum Example 189 (failure)
- Reference: A US footwear company has sparked outrage among consumers and campaign groups over a range of high heeled shoes for babies.
- Top-1: A US company has come under fire for posting images of babies in high heels on its Facebook page.
- Agreement-gated: A US company has come under fire for promoting a range of "high-fashion" shoes for babies that feature babies in high heels.
- Top-1 scores: FactCC=0.9863, NLI=0.0283, SummaC=0.2199
- Agreement-gated scores: FactCC=0.0349, NLI=0.0406, SummaC=0.2330
- Gate reason: fallback_weighted_sum

### xsum Example 197 (failure)
- Reference: Co-operative Bank is cutting 200 jobs as it looks to continue its recovery.
- Top-1: The Co-operative Bank has announced plans to cut up to 1,000 jobs as part of a cost-cutting drive.
- Agreement-gated: The Co-operative Bank has announced plans to cut up to 1,000 jobs as it continues to make losses.
- Top-1 scores: FactCC=0.9257, NLI=0.0014, SummaC=0.2133
- Agreement-gated scores: FactCC=0.0157, NLI=0.0029, SummaC=0.2161
- Gate reason: metric_agreement

## cnn_dailymail

### cnn_dailymail Example 246 (improvement)
- Reference: Maria Shriver's father was stricken by Alzheimer's, a growing scourge in U.S.
Women are disproportionately affected as sufferers and caregivers, she says .
Wipe Out Alzheimer's Challenge is launching to fill in for lagging government funding, she says .
- Top-1: Every 67 seconds, another one of us develops Alzheimer's, says Maria Shriver. Women in their 60s are about twice as likely to develop Alzheimer's as breast cancer, she says.
- Agreement-gated: Every 67 seconds, another one of us develops Alzheimer's. Women in their 60s are about twice as likely to develop Alzheimer's as breast cancer.
- Top-1 scores: FactCC=0.0007, NLI=0.0023, SummaC=0.2984
- Agreement-gated scores: FactCC=1.0000, NLI=0.9812, SummaC=0.3084
- Gate reason: metric_agreement

### cnn_dailymail Example 252 (improvement)
- Reference: The album will feature a 12-minute acoustic Cobain unheard track .
The doc is already winning rave reviews .
Filmmaker wants to release one of the Cobain's personal cassettes .
- Top-1: The accompanying album will feature "a mind-blowing 12-minute acoustic Cobain unheard track" Brett Morgen didn't share any other details regarding the song other than it will feature on the "Montage of Heck" soundtrack.
- Agreement-gated: The accompanying album will feature "a mind-blowing 12-minute acoustic Cobain unheard track," director Brett Morgen tweeted. The HBO documentary "Kurt Cobain: Montage of Heck" is already drawing rave reviews.
- Top-1 scores: FactCC=0.0007, NLI=0.0037, SummaC=0.2427
- Agreement-gated scores: FactCC=0.9997, NLI=0.9859, SummaC=0.2446
- Gate reason: metric_agreement

### cnn_dailymail Example 43 (improvement)
- Reference: Timothy Stanley: GOP senators' letter to Iranian leaders seems extraordinary .
But undermining a president's foreign policy is not at all unique, Stanley says .
He says both left, right have gone around administrations to deal with foreign leaders .
- Top-1: 47 senators have written an open letter to the Iranian regime to advise that any deal agreed to with Obama could be reversed after the 2016 presidential election. Julian Zelizer: People in both parties have done far more remarkable things in the past.
- Agreement-gated: 47 senators have written an open letter to the Iranian regime to advise that any deal agreed to with Obama could be reversed after the 2016 presidential election.
- Top-1 scores: FactCC=0.0000, NLI=0.5032, SummaC=0.2508
- Agreement-gated scores: FactCC=0.9995, NLI=0.9866, SummaC=0.2455
- Gate reason: metric_agreement

### cnn_dailymail Example 278 (failure)
- Reference: The rumors are true: "X-Files" is returning .
David Duchovny, Gillian Anderson and producer Chris Carter are all back .
Carter: "The world has only gotten that much stranger" since the show went off the air in 2002 .
- Top-1: David Duchovny and Gillian Anderson are both back to play Fox Mulder and Dana Scully. "The X-Files" concerned Mulder, an FBI agent who believes in paranormal phenomena.
- Agreement-gated: "The X-Files" ran for nine seasons in the '90s and early '00s. David Duchovny and Gillian Anderson are both back to play Fox Mulder and Dana Scully.
- Top-1 scores: FactCC=0.9995, NLI=0.9934, SummaC=0.2248
- Agreement-gated scores: FactCC=0.0019, NLI=0.9960, SummaC=0.2321
- Gate reason: metric_agreement

### cnn_dailymail Example 186 (failure)
- Reference: Richard III's remains were found beneath a car park in Leicester in 2012 .
Long-lost King's skeleton is to be reinterred in the city's cathedral later this week .
Bones will be buried in a coffin made by Richard III's descendant, Michael Ibsen .
- Top-1: Cabinet-maker Michael Ibsen has just put the finishing touches to a coffin. The casket will be the final resting place of Richard III, who died more than 500 years ago. His DNA was used to establish the identity of the English King, found buried beneath a car parking lot.
- Agreement-gated: Cabinet-maker Michael Ibsen has just put the finishing touches to a coffin. The casket will be the final resting place of Richard III, who died more than 500 years ago. DNA was used to establish the identity of the English King, found buried beneath a car parking lot in Leicester.
- Top-1 scores: FactCC=0.9962, NLI=0.9812, SummaC=0.3619
- Agreement-gated scores: FactCC=0.0083, NLI=0.9898, SummaC=0.3702
- Gate reason: metric_agreement

### cnn_dailymail Example 121 (failure)
- Reference: Ben Stiller and Owen Wilson reprised their roles as male models at Paris Fashion Week .
They walked in the Valentino show as Derek Zoolander and Hansel McDonald .
- Top-1: Ben Stiller and Owen Wilson reprised their roles as the vacuous models from "Zoolander" at Tuesday's women's couture show. There was no mistaking Stiller for anyone other than the fierce Zoolander.
- Agreement-gated: Ben Stiller and Owen Wilson reprised their roles as the vacuous models from the popular 2001 film "Zoolander" The duo is gearing up for " Zoolander 2," which is slated for release in February 2016.
- Top-1 scores: FactCC=0.9958, NLI=0.9324, SummaC=0.2085
- Agreement-gated scores: FactCC=0.9999, NLI=0.0025, SummaC=0.2160
- Gate reason: metric_agreement
