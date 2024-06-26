import PIL.Image
import pygame
import pygame_gui
import sys
import os.path
import random
import numpy as np
import cv2 as cv
import PIL
import datetime
import tkinter as tk
from tkinter import filedialog
from Utils import *
from Network import Network_running


class Game:
    def __init__(self):
        self.interactable = True
        self.tiles = [0 for x in range(784)]
        self.offset = (block_size // 2, block_size // 2)
        self.menu_surf = pygame.Surface((221, 728))
        self.predict = [0 for x in range(10)]
        self.Network = Network_running()
        self.Network.load_model('./models/new.npz')

        menu_surf_rect = self.menu_surf.get_rect()
        text_size = menu_surf_rect.width - 10

        button_load_rect = pygame.Rect(29 * block_size + menu_surf_rect.width // 2 - text_size // 2, 210, text_size, 50)
        self.button_load = pygame_gui.elements.UIButton(button_load_rect, 'Load Image', manager)

        button_save_rect = pygame.Rect(29 * block_size + menu_surf_rect.width // 2 - text_size // 2, 270, text_size, 50)
        self.button_save = pygame_gui.elements.UIButton(button_save_rect, 'Save Image', manager)

        text = "<font face=HomeVideo size=4.5><b>C</b>: To Clear all.\n<b>LEFT MOUSE</b>: To Draw\n<b>RIGHT MOUSE</b>: To Delete</font>"
        text_rect = pygame.Rect(29 * block_size + menu_surf_rect.width // 2 - text_size // 2, 330, text_size, -1)
        text_box = pygame_gui.elements.UITextBox(text, text_rect, manager=manager, wrap_to_height=True)
        text_box.set_active_effect(pygame_gui.TEXT_EFFECT_TYPING_APPEAR, params={'time_per_letter': 0.01})

        self.file_dialog = None
        self.save_dialog = None
        self.confirm_dialog = None

        # file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

    def predicting(self):
        img = np.array(self.tiles)
        self.Network.load_image_np(img)
        res = self.Network.run()
        res = np.round(res * 100, 0).tolist()
        self.predict = [x[0] for x in res]

    def run(self):
        mouse_pos = pygame.mouse.get_pos()
        self.handle_mouse(mouse_pos)
        self.render()

    def render(self):
        self.draw_title()
        self.draw_UI()

    def draw_UI(self):
        # Blit objects on Surface
        self.menu_surf.fill('#4D4D4D')
        for i in range(10):

            rect = pygame.Rect(i * 20 + 11, 150 - self.predict[i], 15, self.predict[i])
            pygame.draw.rect(self.menu_surf, 'gray', rect)

            num = reg_font.render(str(i), True, 'white')
            num_rec = num.get_rect(center=(rect.centerx, rect.bottom + 20))
            self.menu_surf.blit(num, num_rec)

        menu_surf_rect = self.menu_surf.get_rect()
        predict = reg_font.render('Model Predict: ' + str(np.argmax(self.predict)), True, 'white')
        predict_rect = predict.get_rect(center=(menu_surf_rect.centerx, menu_surf_rect.bottom - 45))
        self.menu_surf.blit(predict, predict_rect)

        # Blit the surface on Screen
        Screen.blit(self.menu_surf, (29 * block_size, block_size // 2))

    def handle_mouse(self, mouse_pos):
        global block_size, mouse_clicked
        translate_point = ((mouse_pos[0] - self.offset[0]) // block_size, (mouse_pos[1] - self.offset[1]) // block_size)
        mouse_click = pygame.mouse.get_pressed()
        mouse_released = pygame.mouse.get_just_released()

        if self.interactable:
            if mouse_clicked == False:
                if mouse_click[0]:
                    if translate_point[0] >= 0 and translate_point[0] < 28:
                        if translate_point[1] >= 0 and translate_point[1] < 28:
                            translate_cord = translate_point[1] * 28 + translate_point[0]
                            offset_x = random.choice([-1, 1])
                            offset_y = random.choice([-1, 1])
                            self.tiles[translate_cord] = min(random.randint(120, 180) + self.tiles[translate_cord], 255)

                            # UNCOMMENT THIS FOR NATURAL BRUSH
                            translate_cord_X_0 = translate_point[1] * 28 + translate_point[0] + offset_x
                            translate_cord_X_Y = (translate_point[1] + offset_y) * 28 + translate_point[0] + offset_x
                            translate_cord_0_Y = (translate_point[1] + offset_y) * 28 + translate_point[0]

                            if translate_cord_X_0 >= 0 and translate_cord_X_0 < 783:
                                self.tiles[translate_cord_X_0] = min(random.randint(0, 30) + self.tiles[translate_cord_X_0], 255)

                            if translate_cord_X_Y >= 0 and translate_cord_X_Y < 783:
                                self.tiles[translate_cord_X_Y] = min(random.randint(0, 30) + self.tiles[translate_cord_X_Y], 255)

                            if translate_cord_0_Y >= 0 and translate_cord_0_Y < 783:
                                self.tiles[translate_cord_0_Y] = min(random.randint(0, 30) + self.tiles[translate_cord_0_Y], 255)

                            mouse_clicked = True
                            self.predicting()

                elif mouse_click[2]:
                    if translate_point[0] >= 0 and translate_point[0] < 28:
                        if translate_point[1] >= 0 and translate_point[1] < 28:
                            translate_cord = translate_point[1] * 28 + translate_point[0]
                            self.tiles[translate_cord] = 0
                            mouse_clicked = True
                            self.predicting()

            # ENABLE DRAGGING
            movement = pygame.mouse.get_rel()
            if movement[0] != 0 or movement[1] != 0:
                mouse_clicked = False

            if mouse_clicked:
                if mouse_released[0] or mouse_released[2]:
                    mouse_clicked = False

    def draw_title(self):
        global block_size
        x = 0
        y = -1
        for i in range(784):
            x = i % 28
            if x == 0:
                y += 1
            rect = pygame.Rect(x * block_size + self.offset[0], y * block_size + self.offset[1], block_size, block_size)
            pixel = min(self.tiles[i], 255)
            color = (pixel, pixel, pixel)
            pygame.draw.rect(Screen, color, rect)

    def clear(self):
        self.tiles = [0 for x in range(784)]

    def set_button_state(self, bool):
        if bool:
            self.button_load.enable()
            self.button_save.enable()
            self.interactable = True
        else:
            self.button_load.disable()
            self.button_save.disable()
            self.interactable = False

    def load_image(self, path):
        try:
            image = np.array(PIL.Image.open(path))
            image = cv.resize(image, (28, 28))
            image = image.mean(axis=2)
            image = image.reshape((784, 1))
            image = image.astype('uint8')
            image = image.tolist()
            self.tiles = [min(x[0], 255) for x in image]
            # print(self.tiles)
            self.predicting()
        except Exception as error:
            print('An error occurred:', error)

    def save_image(self, path):
        try:
            array = []
            for i in self.tiles:
                array.append(i)
                array.append(i)
                array.append(i)

            array = np.array(array)
            array = array.reshape((28, 28, 3)).astype('uint8')
            image = PIL.Image.fromarray(array)
            # path = os.path.relpath(path)
            image.save(path)
        except Exception as error:
            print('An error occurred:', error)


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption('Simple Digit Recognition')
    # root = tk.Tk()
    # root.withdraw()
    block_size = 26
    Screen_w = 988
    Screen_h = 754
    Screen = pygame.display.set_mode((Screen_w, Screen_h))
    mouse_clicked = False
    manager = pygame_gui.UIManager((Screen_w, Screen_h), 'theme/theme.json')
    reg_font = pygame.Font('fonts/HomeVideo-BLG6G.ttf', 20)
    clock = pygame.time.Clock()

    # FONTS
    manager.add_font_paths("HomeVideo", "fonts/HomeVideo-BLG6G.ttf", bold_path="fonts/PixeloidSansBold-PKnYd.ttf")
    manager.preload_fonts([{'name': 'HomeVideo', 'point_size': 14, 'style': 'regular'},
                           {'name': 'HomeVideo', 'point_size': 14, 'style': 'bold'},
                           {'name': 'HomeVideo', 'point_size': 12, 'style': 'regular'},
                           {'name': 'HomeVideo', 'point_size': 12, 'style': 'bold'},
                           {'name': 'HomeVideo', 'point_size': 16, 'style': 'regular'},
                           {'name': 'HomeVideo', 'point_size': 16, 'style': 'bold'},
                           {'name': 'HomeVideo', 'point_size': 18, 'style': 'regular'},
                           {'name': 'HomeVideo', 'point_size': 18, 'style': 'bold'}])

    game = Game()

    while True:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                # root.destroy()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    if game.interactable:
                        game.clear()

            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == game.button_load:
                    game.file_dialog = pygame_gui.windows.UIFileDialog(pygame.Rect(Screen_w // 2 - 220, Screen_h // 2 - 250, 440, 500),
                                                                       manager,
                                                                       window_title='Load Image...',
                                                                       initial_file_path='images/',
                                                                       allow_picking_directories=False,
                                                                       allow_existing_files_only=True,
                                                                       allowed_suffixes={".png", ".jpg"})
                    game.set_button_state(False)

                if event.ui_element == game.button_save:
                    text = str(datetime.datetime.now()).split(' ')[0] + '-' + str(random.randint(1, 600)) + '.png'
                    game.save_dialog = pygame_gui.windows.UIFileDialog(pygame.Rect(Screen_w // 2 - 220, Screen_h // 2 - 250, 440, 500),
                                                                       manager,
                                                                       window_title='Save Image...',
                                                                       initial_file_path='images/' + text,
                                                                       allow_picking_directories=False,
                                                                       allowed_suffixes={".png", ".jpg"})
                    game.set_button_state(False)

            if event.type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                if event.ui_element == game.file_dialog:
                    # print(game.file_dialog.current_file_path)  # OR event.text
                    if (event.text != None):
                        game.load_image(event.text)

                if event.ui_element == game.save_dialog:
                    # print(event.text)
                    if os.path.isfile(event.text):
                        game.confirm_dialog = pygame_gui.windows.UIConfirmationDialog(pygame.Rect(Screen_w // 2 - 220, Screen_h // 2 - 100, 440, 200),
                                                                                      action_long_desc='Do you want to replace the existing file?',
                                                                                      manager=manager,
                                                                                      window_title='Replace File',
                                                                                      action_short_name='Replace')
                    else:
                        game.save_image(event.text)

            if event.type == pygame_gui.UI_CONFIRMATION_DIALOG_CONFIRMED:
                if event.ui_element == game.confirm_dialog:
                    game.save_image(game.save_dialog.current_file_path)

            if event.type == pygame_gui.UI_WINDOW_CLOSE:
                if event.ui_element == game.file_dialog:
                    game.set_button_state(True)

                if event.ui_element == game.save_dialog:
                    if game.confirm_dialog == None:
                        game.set_button_state(True)

                if event.ui_element == game.confirm_dialog:
                    game.set_button_state(True)
                    game.confirm_dialog = None

            manager.process_events(event)

        manager.update(time_delta)
        Screen.fill('#191B1F')
        game.run()
        manager.draw_ui(Screen)
        pygame.display.flip()
