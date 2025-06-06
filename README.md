# LOKĀLI DARBINĀMO LIELO VALODU MODEĻU APSKATS
## Bakalaura darbs
## Autore: Anastasija Ostrovska

# Anotācija 
Strauji attīstošie lielu valodu modeļi (LVM) ne tikai sniedz jaunas iespējas teksta prognozēšanā, apstrādē un ģenerēšanā, bet arī rada aizdomas par datu privātumu, jo lielākā daļa LVM ir balstīti mākonī. Tāpēc radās pieprasījums pēc lokālas LVM izmantošanas, saglabājot gan  modeļu veiktspēju, gan datu konfidencialitāti. Lokāla LVM izmantošana dod iespēju veikt to apmācību uz specifiskām datu kopām, kas pieder lietotājam vai organizācijai, tādējādi uzlabojot modeļa veiktspējas precizitāti, kā arī nodrošinot tā pielāgošanu konkrētām vajadzībām. Pētījuma mērķis ir salīdzināt un novērtēt lokāli darbināmos LVM un to pielietojumus. Lai sasniegtu mērķi, tika veikti divu klašu eksperimenti: videokaršu specifikāciju izpēte un modeļu apmācības iespēju novērtējums dabiskās valodas apstrādes uzdevumos - vēstuļu klasifikācijā un sarunas kopsavilkumu veidošanā. Eksperimentiem tika izvēlēti trīs modeļi – Mistral ar 7 miljardu parametriem, Phi-2 ar 2.7 miljardu parametriem un TinyLLaMA ar 1.1 miljardu parametru. Tika novērtēti lokālo LVM izmantošanas resursu ierobežojumi. Mistral modelis ir piemērots uzdevumiem, kuriem ir nepieciešama lielāka precizitāte un ir pieejami vismaz 12 GB videokartes atmiņas ar CUDA atbalstu, TinyLLaMA modelis ir lietojams uzdevumiem ar ierobežotiem skaitļošanas resursiem, piemēram, videokartēm ar 4 GB CUDA atmiņu. Noskaidrots, ka mazākie modeļi, izmantojot precīzo ieregulēšanu, sasniedz konkurētspējīgu veiktspēju ar mazāku apmācības laiku un skaitļošanas resursu patēriņu. Darbs sniedz praktisku ieguldījumu, piedāvājot instrukciju lokālo LVM izmantošanai, kas dod iespēju saglabāt datu privātumu un pielāgot modeļus specifiskiem uzdevumiem. 

# Saglābātie faili 
- 3_dala_uzdevumi.py, Tas ir izmantojams skaitļošanas resursu noteikšanai 3.4. sadaļā — eksperimentu veikšanai ar GPU bez CUDA atbalsta. Šajā laikā tika noskaidrots, ka modeļus iespējams izmantot arī ierobežotu skaitļošanas resursu apstākļos.
- prasibu_saraksts.txt - Ir aprakstītas nepieciešamo *Python* pakotņu versijas, kas vajadzīgas kopsavilkuma veidošanas un klasifikācijas uzdevumu apmācībai.
- 4_sadala_klasifikacijas_uzdevums.py - Tas ir izmantots 3.5. eksperimentā resursu noskaidrošanai un 4.3. sadaļā klasifikācijas uzdevumu apmācībai.
- 4_sadala_kopsavilkums_uzdevums.py - Tas ir izmantots 4.4. sadaļā kopsavilkuma veidošanas uzdevuma apmācībai.

# Informācijas avoti
- Python. (2025) Python. Python dokumentācija. Pieejams: https://docs.python.org/3.12/ 
- De Silva, M. T. (2023). Multilingual Dialogue Summary Generation System for Customer Services (Doctoral dissertation, University of Westminster). Pieejams: https://doi.org/10.13140/RG.2.2.34689.02400
- Massaron, L. Fine-tune Llama 2 for sentiment analysis. Python kods. Kaggle, 2024. Pieejams: https://www.kaggle.com/code/lucamassaron/fine-tune-llama-2-for-sentiment-analysis/ [skatīts: 22.03.2025]
- HuggingFace. HuggingFace: The AI community building the future (2025). Pieejams: https://huggingface.co [skatīts: 21.03.2025]
