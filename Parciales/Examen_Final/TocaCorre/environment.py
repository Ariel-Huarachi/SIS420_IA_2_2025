import random
import numpy as np

from agents import Cazador
from maze import Maze
from rewards import calcular_recompensas_3agentes
from step_logic import (
    mover_agente,
    aplicar_zonas,
    detectar_toca_3agentes,
    aplicar_trampa_3agentes,
    detectar_atasco_global,
    desatascar_agentes
)

class TocaCorreContinuousEnv3Agentes:

    def __init__(self):
        self.maze = Maze()

        # radio por defecto de los agentes
        self.agent_radius = 0.3

        # velocidades base
        self.speed_base = 1.0

        # zonas especiales
        self.zonas_acelerar = [(5, 5, 1.5), (12, 10, 1.5)]
        self.zonas_lentas = [(8, 3, 0.5), (15, 15, 0.6)]

        # 3 agentes en ciclo: A->B, B->C, C->A
        self.agentes = []  # [A, B, C]

        self.done = False
        self.steps = 0
        
        # Estadísticas del episodio
        self.capturas_totales = [0, 0, 0]  # Cada agente como cazador
        self.total_distancia = [0, 0, 0]

    # ==========================================================
    #   POSICIÓN ALEATORIA NO EN PAREDES
    # ==========================================================
    def _random_valid_position(self):
        max_intentos = 100
        for _ in range(max_intentos):
            x = random.uniform(1, self.maze.width - 1)
            y = random.uniform(1, self.maze.height - 1)
            if not self.maze.check_collision(x, y, self.agent_radius):
                return (x, y)
        return (self.maze.width / 2, self.maze.height / 2)

    # ==========================================================
    #   RESET DEL ENTORNO
    # ==========================================================
    def reset(self):
        # Crear 3 agentes en posiciones diferentes
        posiciones = []
        for _ in range(3):
            pos = self._random_valid_position()
            # Asegurar distancia mínima entre agentes
            intentos = 0
            while any(np.linalg.norm([pos[0] - p[0], pos[1] - p[1]]) < 3.0 
                     for p in posiciones) and intentos < 50:
                pos = self._random_valid_position()
                intentos += 1
            posiciones.append(pos)

        self.agentes = []
        for i, (x, y) in enumerate(posiciones):
            agente = Cazador(x, y, id=i, radius=self.agent_radius, speed=self.speed_base)
            self.agentes.append(agente)

        # Resetear estadísticas
        self.steps = 0
        self.done = False
        self.capturas_totales = [0, 0, 0]
        self.total_distancia = [0, 0, 0]

        return self._get_state()

    # ==========================================================
    #   OBTENER ESTADO (PARA RL)
    # ==========================================================
    def _get_state(self):
        """
        Estado para cada agente:
        - Posición propia (x, y)
        - Posición del agente que persigue (x, y)
        - Posición del agente que lo persigue (x, y)
        - Distancia a presa
        - Ángulo a presa
        - Tiene trampa activa
        - Distancia al perseguidor
        """
        estados = []
        
        for i in range(3):
            cazador = self.agentes[i]
            presa = self.agentes[(i + 1) % 3]  # Siguiente en ciclo
            perseguidor = self.agentes[(i - 1) % 3]  # Anterior en ciclo
            
            dx = presa.x - cazador.x
            dy = presa.y - cazador.y
            distancia = np.linalg.norm([dx, dy])
            angulo = np.arctan2(dy, dx)
            
            # Distancia al perseguidor
            dx_pers = perseguidor.x - cazador.x
            dy_pers = perseguidor.y - cazador.y
            dist_pers = np.linalg.norm([dx_pers, dy_pers])
            
            estado = np.array([
                cazador.x,
                cazador.y,
                presa.x,
                presa.y,
                distancia,
                angulo,
                1 if cazador.trampa_activa is not None else 0,
                dist_pers
            ], dtype=float)
            
            estados.append(estado)
        
        return estados

    # ==========================================================
    #   STEP – LÓGICA COMPLETA CON ANTI-ATASCOS
    # ==========================================================
    def step(self, acciones):
        """
        acciones: lista de 3 acciones (una por agente)
        """
        self.steps += 1
        done = False

        prev_posiciones = [agente.get_position() for agente in self.agentes]

        # 1. Mover cada agente
        for i, agente in enumerate(self.agentes):
            accion = acciones[i]
            if accion == 8:
                agente.poner_trampa()
            else:
                mover_agente(agente, accion, self.maze)

        # 2. Registrar distancias
        for i, agente in enumerate(self.agentes):
            nueva_pos = agente.get_position()
            dist = np.linalg.norm(np.array(nueva_pos) - np.array(prev_posiciones[i]))
            self.total_distancia[i] += dist

        # 3. Aplicar trampas y zonas
        capturas = aplicar_trampa_3agentes(self)
        for agente in self.agentes:
            aplicar_zonas(agente, self)

        # 4. Detección de atascos
        if self.steps % 20 == 0:
            if detectar_atasco_global(self.agentes):
                desatascar_agentes(self.agentes)

        # 5. Detectar toques
        toques = detectar_toca_3agentes(self.agentes)
        for i, toco in enumerate(toques):
            if toco:
                self.capturas_totales[i] += 1

        # 6. Calcular recompensas
        recompensas = calcular_recompensas_3agentes(
            self, prev_posiciones, toques, capturas
        )

        estado_nuevo = self._get_state()

        return estado_nuevo, recompensas, done

    # ==========================================================
    #   ESTADÍSTICAS
    # ==========================================================
    def get_stats(self):
        return {
            'steps': self.steps,
            'capturas_total': sum(self.capturas_totales),
            'capturas_por_agente': self.capturas_totales.copy(),
            'distancia_total': sum(self.total_distancia),
            'distancia_por_agente': self.total_distancia.copy(),
        }