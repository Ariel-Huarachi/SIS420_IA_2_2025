import pygame
import numpy as np
from environment import TocaCorreContinuousEnv
from train import discretizar_estado
import math

# COLORES
COLOR_PER = (255, 107, 107)
COLOR_EVA = (78, 205, 196)
COLOR_TRAMPA = (155, 89, 182)
COLOR_ZONA_RAPIDA = (46, 204, 113)
COLOR_ZONA_LENTA = (243, 156, 18)
COLOR_PARED = (236, 240, 241)
COLOR_FONDO = (44, 62, 80)
COLOR_PANEL = (28, 40, 51)
COLOR_SALTO = (255, 200, 0)

# Dimensiones
WIN_W = 1400
WIN_H = 800
PANEL_W = 350
JUEGO_W = 700
LEGEND_W = 350

# Inicializar Pygame
pygame.init()
pygame.display.set_caption("🎮 Toca y Corre - RL | Ariel Huarachi")

screen = pygame.display.set_mode((WIN_W, WIN_H))
clock = pygame.time.Clock()

FONT_SMALL = pygame.font.SysFont("monospace", 16)
FONT_MED = pygame.font.SysFont("monospace", 18)
FONT_BIG = pygame.font.SysFont("monospace", 22, bold=True)
FONT_TITLE = pygame.font.SysFont("monospace", 24, bold=True)


def draw_text(surface, text, x, y, color=(255,255,255), font=FONT_SMALL):
    for line in text.split("\n"):
        surface.blit(font.render(line, True, color), (x, y))
        y += font.get_linesize()
    return y


def map_to_game(x, y, maze_w, maze_h):
    scale_x = JUEGO_W / maze_w
    scale_y = JUEGO_W / maze_h
    px = PANEL_W + int(x * scale_x)
    py = int(y * scale_y)
    return px, py


def draw_walls(env):
    for (x1, y1), (x2, y2) in env.maze.walls:
        px1, py1 = map_to_game(x1, y1, env.maze.width, env.maze.height)
        px2, py2 = map_to_game(x2, y2, env.maze.width, env.maze.height)
        pygame.draw.line(screen, COLOR_PARED, (px1, py1), (px2, py2), 4)


def draw_zones(env):
    for zx, zy, _ in env.zonas_acelerar:
        px, py = map_to_game(zx, zy, env.maze.width, env.maze.height)
        pygame.draw.circle(screen, COLOR_ZONA_RAPIDA, (px, py), 25)

    for zx, zy, _ in env.zonas_lentas:
        px, py = map_to_game(zx, zy, env.maze.width, env.maze.height)
        pygame.draw.circle(screen, COLOR_ZONA_LENTA, (px, py), 25)


def draw_agent(agent, color, env):
    px, py = map_to_game(agent.x, agent.y, env.maze.width, env.maze.height)
    radio = int(agent.radius * 20)

    pygame.draw.circle(screen, (0, 0, 0), (px+3, py+3), radio)
    pygame.draw.circle(screen, color, (px, py), radio)
    pygame.draw.circle(screen, (255, 255, 255), (px, py), radio, 2)


def draw_trampa(env):
    if env.evasor.trampa_activa:
        tx, ty = env.evasor.trampa_activa
        px, py = map_to_game(tx, ty, env.maze.width, env.maze.height)
        
        size = 12 + int(4 * math.sin(pygame.time.get_ticks() / 200))
        pygame.draw.circle(screen, COLOR_TRAMPA, (px, py), size)
        pygame.draw.circle(screen, (255, 255, 255), (px, py), size, 2)


def draw_salto_indicator(env):
    if env.perseguidor.salto_disponible:
        px, py = map_to_game(env.perseguidor.x, env.perseguidor.y, 
                            env.maze.width, env.maze.height)
        pygame.draw.circle(screen, COLOR_SALTO, (px, py - 25), 8)
        text = FONT_SMALL.render("JUMP", True, COLOR_SALTO)
        screen.blit(text, (px - 20, py - 40))


def draw_trayectories(trayectoria_per, trayectoria_eva, env):
    if len(trayectoria_per) > 1:
        pts = [map_to_game(x, y, env.maze.width, env.maze.height)
               for x, y in trayectoria_per[-100:]]
        if len(pts) > 1:
            pygame.draw.lines(screen, COLOR_PER, False, pts, 2)

    if len(trayectoria_eva) > 1:
        pts = [map_to_game(x, y, env.maze.width, env.maze.height)
               for x, y in trayectoria_eva[-100:]]
        if len(pts) > 1:
            pygame.draw.lines(screen, COLOR_EVA, False, pts, 2)


def draw_panel(env, episodio, pasos, step_count, intercambios, distancia):
    pygame.draw.rect(screen, COLOR_PANEL, (0, 0, PANEL_W, WIN_H))

    y = 15
    y = draw_text(screen, "🎮 TOCA Y CORRE", 20, y, COLOR_EVA, FONT_TITLE)
    y = draw_text(screen, "Aprendizaje RL", 20, y, (180, 180, 180), FONT_SMALL)
    y += 15
    
    pygame.draw.line(screen, (100, 100, 100), (20, y), (PANEL_W-20, y), 2)
    y += 15

    y = draw_text(screen, "📊 ESTADÍSTICAS", 20, y, (255, 255, 255), FONT_BIG)
    y += 5

    panel_text = f"""
Episodio: {episodio}
Paso: {step_count} / {pasos}
Distancia: {distancia:.2f} u
Intercambios: {intercambios}
""".strip()
    y = draw_text(screen, panel_text, 20, y, (220, 220, 220), FONT_MED)
    y += 20

    pygame.draw.line(screen, (100, 100, 100), (20, y), (PANEL_W-20, y), 2)
    y += 15

    y = draw_text(screen, "🔴 PERSEGUIDOR", 20, y, COLOR_PER, FONT_BIG)
    y += 5
    
    per_text = f"""
Posición: ({env.perseguidor.x:.1f}, {env.perseguidor.y:.1f})
Velocidad: {env.perseguidor.speed:.2f}
Salto: {"✅ SÍ" if env.perseguidor.salto_disponible else "❌ NO"}
Saltos: {env.perseguidor.saltos_usados}
Dist. total: {env.total_distancia_recorrida_per:.1f}
""".strip()
    y = draw_text(screen, per_text, 20, y, (220, 220, 220), FONT_SMALL)
    y += 20

    pygame.draw.line(screen, (100, 100, 100), (20, y), (PANEL_W-20, y), 2)
    y += 15

    y = draw_text(screen, "🔵 EVASOR", 20, y, COLOR_EVA, FONT_BIG)
    y += 5
    
    eva_text = f"""
Posición: ({env.evasor.x:.1f}, {env.evasor.y:.1f})
Velocidad: {env.evasor.speed:.2f}
Trampa: {"✅ SÍ" if env.evasor.trampa_activa else "❌ NO"}
Trampas: {env.evasor.trampas_puestas}
Dist. total: {env.total_distancia_recorrida_eva:.1f}
""".strip()
    y = draw_text(screen, eva_text, 20, y, (220, 220, 220), FONT_SMALL)
    y += 20

    pygame.draw.line(screen, (100, 100, 100), (20, y), (PANEL_W-20, y), 2)
    y += 15

    y = draw_text(screen, "👨‍💻 Ariel Huarachi", 20, y, (150, 150, 150), FONT_SMALL)


def draw_legend():
    pygame.draw.rect(screen, COLOR_PANEL, (PANEL_W + JUEGO_W, 0, LEGEND_W, WIN_H))

    y = 15
    y = draw_text(screen, "🎯 ELEMENTOS", PANEL_W + JUEGO_W + 20, y, 
                  (255, 255, 255), FONT_BIG)
    y += 15

    legend_items = [
        ("🔴 Perseguidor", COLOR_PER),
        ("🔵 Evasor", COLOR_EVA),
        ("🟣 Trampa", COLOR_TRAMPA),
        ("🟢 Zona Rápida", COLOR_ZONA_RAPIDA),
        ("🟠 Zona Lenta", COLOR_ZONA_LENTA),
        ("⬜ Pared", COLOR_PARED),
        ("⭐ Salto", COLOR_SALTO)
    ]

    for name, color in legend_items:
        pygame.draw.circle(screen, color,
                          (PANEL_W + JUEGO_W + 30, y + 15), 10)
        draw_text(screen, name, PANEL_W + JUEGO_W + 50, y + 8, 
                 (220, 220, 220), FONT_SMALL)
        y += 35

    y += 20
    pygame.draw.line(screen, (100, 100, 100), 
                    (PANEL_W + JUEGO_W + 20, y), 
                    (WIN_W - 20, y), 2)
    y += 20

    y = draw_text(screen, "⌨️  CONTROLES", PANEL_W + JUEGO_W + 20, y, 
                  (255, 255, 255), FONT_BIG)
    y += 10

    controls = """
ESC - Salir
ESPACIO - Pausar
R - Reiniciar
""".strip()
    draw_text(screen, controls, PANEL_W + JUEGO_W + 20, y, 
             (200, 200, 200), FONT_SMALL)


def visualizar_pygame(Q_per, Q_eva, episodios=1, pasos=500, 
                     mostrar_trayectorias=True, fps=60):
    """
    🔥 VISUALIZACIÓN CON DIAGNÓSTICO
    """
    
    # ===== DIAGNÓSTICO DEL MODELO =====
    print("\n" + "="*70)
    print("  🔍 DIAGNÓSTICO DEL MODELO")
    print("="*70)
    print(f"  Estados en Q_per: {len(Q_per)}")
    print(f"  Estados en Q_eva: {len(Q_eva)}")
    
    if len(Q_per) > 0:
        estado_ejemplo = list(Q_per.keys())[0]
        valores = Q_per[estado_ejemplo]
        suma = np.sum(np.abs(valores))
        
        print(f"\n  Ejemplo Q_per[{estado_ejemplo}]:")
        print(f"    Valores: {valores}")
        print(f"    Suma absoluta: {suma:.4f}")
        print(f"    Max acción: {np.argmax(valores)}")
        
        if suma < 0.001:
            print("\n  ⚠️  ADVERTENCIA: Modelo no entrenado o vacío!")
            print("     Por favor entrena primero (opción 1 del menú)")
            return
        else:
            print("\n  ✅ Modelo parece entrenado correctamente")
    else:
        print("\n  ❌ ERROR: Q-tables vacías")
        print("     Por favor entrena primero (opción 1 del menú)")
        return
    
    print("="*70 + "\n")
    
    # ===== INICIAR VISUALIZACIÓN =====
    env = TocaCorreContinuousEnv()
    episodio_actual = 1
    step_count = 0
    intercambios = 0
    paused = False

    estado = env.reset()
    trayectoria_per = []
    trayectoria_eva = []

    running = True
    while running:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    estado = env.reset()
                    trayectoria_per = []
                    trayectoria_eva = []
                    step_count = 0
                    intercambios = 0

        if paused:
            text = FONT_TITLE.render("⏸️ PAUSADO", True, (255, 255, 0))
            screen.blit(text, (PANEL_W + JUEGO_W//2 - 80, WIN_H//2))
            pygame.display.flip()
            continue

        # Seleccionar acciones
        estado_d = discretizar_estado(estado)
        a_per = np.argmax(Q_per[estado_d])
        a_eva = np.argmax(Q_eva[estado_d])

        prev_pos_per = env.perseguidor.get_position()
        prev_pos_eva = env.evasor.get_position()

        # Ejecutar
        estado, r_per, r_eva, done = env.step(a_per, a_eva)
        step_count += 1

        # Detectar intercambio
        nueva_pos_per = env.perseguidor.get_position()
        nueva_pos_eva = env.evasor.get_position()

        if abs(nueva_pos_per[0] - prev_pos_eva[0]) < 0.4 and \
           abs(nueva_pos_per[1] - prev_pos_eva[1]) < 0.4:
            intercambios += 1

        # Distancia
        distancia = np.linalg.norm([
            env.perseguidor.x - env.evasor.x,
            env.perseguidor.y - env.evasor.y
        ])

        # Fin de episodio
        if step_count >= pasos or done:
            episodio_actual += 1
            if episodio_actual > episodios:
                episodio_actual = 1

            estado = env.reset()
            trayectoria_per = []
            trayectoria_eva = []
            step_count = 0
            intercambios = 0
            continue

        # Guardar trayectorias
        trayectoria_per.append((env.perseguidor.x, env.perseguidor.y))
        trayectoria_eva.append((env.evasor.x, env.evasor.y))

        # DIBUJAR
        screen.fill(COLOR_FONDO)

        draw_zones(env)
        draw_walls(env)
        
        if mostrar_trayectorias:
            draw_trayectories(trayectoria_per, trayectoria_eva, env)

        draw_agent(env.perseguidor, COLOR_PER, env)
        draw_agent(env.evasor, COLOR_EVA, env)
        draw_trampa(env)
        draw_salto_indicator(env)

        draw_panel(env, episodio_actual, pasos, step_count, intercambios, distancia)
        draw_legend()

        pygame.display.flip()

    pygame.quit()


def visualizar(Q_per, Q_eva, episodios=1, pasos=500, 
              mostrar_trayectorias=True, velocidad=80):
    fps = max(10, int(1000 / velocidad))
    visualizar_pygame(Q_per, Q_eva, episodios, pasos, 
                     mostrar_trayectorias, fps)


if __name__ == "__main__":
    import numpy as np
    from collections import defaultdict

    print("⚙️ Cargando modelo...")

    try:
        Q_per_dict = np.load("Q_per.npy", allow_pickle=True).item()
        Q_eva_dict = np.load("Q_eva.npy", allow_pickle=True).item()

        Q_per = defaultdict(lambda: np.zeros(9))
        Q_eva = defaultdict(lambda: np.zeros(9))

        Q_per.update(Q_per_dict)
        Q_eva.update(Q_eva_dict)

        print("✅ Modelo cargado")

    except:
        print("❌ Modelo no encontrado")
        Q_per = defaultdict(lambda: np.zeros(9))
        Q_eva = defaultdict(lambda: np.zeros(9))

    visualizar(Q_per, Q_eva, episodios=3, pasos=300, 
              mostrar_trayectorias=True, velocidad=60)

    print("👋 Visualización finalizada")