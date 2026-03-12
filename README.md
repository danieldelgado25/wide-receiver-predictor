# wide-receiver-predictor

Uses statistics, probability, and data analytics to forecast wide receiver fantasy football performance — or at least test how close we can get in a sport driven by chaos, variance, injuries, and small weekly sample sizes.

Predicting fantasy football output is difficult because weekly production is influenced by far more than raw talent alone. Matchups, target competition, quarterback play, injuries, touchdowns, game script, and randomness can all swing results from one week to the next.

This project does not aim to guarantee perfect predictions. The goal is to explore what can realistically be learned from historical NFL data, engineered features, and statistical modeling, with a specific focus on wide receivers. Since WR performance is heavily shaped by volume, efficiency, and situational context, it provides a strong use case for combining sports analytics with machine learning.

## Project Goal

Build a training-ready dataset and prediction pipeline that can estimate future fantasy performance for wide receivers based on past NFL usage and production trends.

The project is centered around:
- collecting historical weekly WR data
- engineering pregame features
- defining a fantasy-relevant target variable
- training and evaluating predictive models
- comparing results against the reality of football variance

## Data Direction

This project will use `nflreadpy` as the main Python data backbone.

Fantasy-specific enrichment libraries from the ffverse ecosystem may also be incorporated later where useful.

Possible future supplements include:
- `ffverse`
- `ffopportunity`

These may help add fantasy-relevant context such as opportunity-based metrics, but the main data workflow will be built around `nflreadpy`.

## Modeling Approach

The planned dataset structure is:

- one row per wide receiver per week
- only features that would be known before the predicted game
- a target such as next-week fantasy points

This setup is intended to avoid data leakage and create a cleaner foundation for machine learning.

Examples of future feature categories may include:
- previous week fantasy points
- rolling averages over recent games (noting greater volatility in certain players)
- targets, receptions, air yards, and touchdowns
- target share and usage trends
- team offensive environment
- opponent defensive context
- game location and other situational factors

## Development Structure

Project logic should live in reusable Python modules under `src/`.

Jupyter notebooks are reserved for:
- experimentation
- feature exploration
- model training
- evaluation
- visualization

That separation keeps the project cleaner and makes it easier to reuse code outside of notebooks.

## Why Wide Receivers?

Wide receivers are one of the most volatile fantasy positions, which makes them both frustrating and interesting to study. A receiver can post a huge week from a handful of targets or disappear despite strong usage. That makes WR prediction a good test bed for exploring:
- probability and uncertainty
- variance in sports performance
- feature engineering
- model limitations in noisy real-world data

## Disclaimer

This project is an analytics and learning tool, not a promise of accurate fantasy outcomes.