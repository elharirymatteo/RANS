FROM nvcr.io/nvidia/isaac-sim:2022.2.1

ARG RANS_WS="/workspace/RANS"
ARG RL_GAMES_WS="/workspace/RANS/rl_games"
ARG RL_GAMES_PATH="rl_games"

ENV RANS_WS=${RANS_WS}
ENV RL_GAMES_WS=${RL_GAMES_WS}
ENV RL_GAMES_PATH=${RL_GAMES_PATH}

COPY ./ ${RANS_WS}

RUN /isaac-sim/python.sh -m pip install --upgrade pip
RUN ls ${RANS_WS}
RUN /isaac-sim/python.sh -m pip install -e $RANS_WS
RUN /isaac-sim/python.sh -m pip install -e $RL_GAMES_WS

WORKDIR ${RANS_WS}/omniisaacgymenvs

ENTRYPOINT ["sh", "-c", "bash"]