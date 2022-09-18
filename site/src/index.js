/* eslint-disable import/prefer-default-export */
/* eslint-disable import/no-extraneous-dependencies */

import '@kitware/vtk.js/favicon';

// Load the rendering pieces we want to use (for both WebGL and WebGPU)
import '@kitware/vtk.js/Rendering/Profiles/All';

import macro from '@kitware/vtk.js/macros';
import DataAccessHelper from '@kitware/vtk.js/IO/Core/DataAccessHelper';
import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkHttpSceneLoader from '@kitware/vtk.js/IO/Core/HttpSceneLoader';
import HttpDataAccessHelper from '@kitware/vtk.js/IO/Core/DataAccessHelper/HttpDataAccessHelper';

// Force DataAccessHelper to have access to various data source
import '@kitware/vtk.js/IO/Core/DataAccessHelper/HtmlDataAccessHelper';
import '@kitware/vtk.js/IO/Core/DataAccessHelper/JSZipDataAccessHelper';

let widgetCreated = false;

function emptyContainer(container) {
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }
}

export function load(container, url) {

    emptyContainer(container);

    const fullScreenRenderer = vtkFullScreenRenderWindow.newInstance({
        background: [1, 1, 1],
        rootContainer: container,
        containerStyle: { height: '100%', width: '100%', position: 'absolute' },
    });
    const renderer = fullScreenRenderer.getRenderer();
    const renderWindow = fullScreenRenderer.getRenderWindow();
    global.renderWindow = renderWindow;

    function onReady(sceneImporter, renderer) {
        sceneImporter.onReady(() => {
            renderer.resetCamera();
            renderWindow.render();
        });

        window.addEventListener('dblclick', () => {
            sceneImporter.resetScene();
            renderWindow.render();
        });
    }

    const progressContainer = document.createElement('div');
    container.appendChild(progressContainer);

    const progressCallback = (progressEvent) => {
        if (progressEvent.lengthComputable) {
            const percent = Math.floor(
                (100 * progressEvent.loaded) / progressEvent.total
            );
            progressContainer.innerHTML = `Loading ${percent}%`;
        } else {
            progressContainer.innerHTML = macro.formatBytesToProperUnit(
                progressEvent.loaded
            );
        }
    };

    HttpDataAccessHelper.fetchBinary(url, {
        progressCallback,
    }).then((zipContent) => {
        container.removeChild(progressContainer);
        const dataAccessHelper = DataAccessHelper.get('zip', {
            zipContent,
            callback: (zip) => {
                const sceneImporter = vtkHttpSceneLoader.newInstance({
                    renderer,
                    dataAccessHelper,
                });
                sceneImporter.setUrl('index.json');
                onReady(sceneImporter, renderer);
            },
        });
    });
}

const exampleContainer = document.querySelector('.content');
const rootBody = document.querySelector('body');
const myContainer = exampleContainer || rootBody;
if (myContainer) {
    //myContainer.classList.add(style.fullScreen);
    rootBody.style.margin = '0';
    rootBody.style.padding = '0';
}

const path = require("./scenes/gps.vtkjs");

load(myContainer, path);